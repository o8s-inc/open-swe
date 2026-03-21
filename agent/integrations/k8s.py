"""Kubernetes pod sandbox backend.

Creates Sandbox CRs (mc.o8s.io/v1alpha1) managed by the agent-controller.
Executes commands via the Kubernetes exec API in the provisioned pod.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

from deepagents.backends.protocol import ExecuteResponse, WriteResult
from deepagents.backends.sandbox import BaseSandbox
from kubernetes import client, config
from kubernetes.stream import stream as k8s_stream

SANDBOX_API_GROUP = "mc.o8s.io"
SANDBOX_API_VERSION = "v1alpha1"
SANDBOX_PLURAL = "sandboxes"


def _load_k8s_config():
    """Load Kubernetes config (in-cluster or kubeconfig)."""
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def _get_config():
    """Read sandbox configuration from environment."""
    return {
        "namespace": os.getenv("SANDBOX_NAMESPACE", "default"),
        "image": os.getenv("SANDBOX_IMAGE", "python:3.12-slim"),
        "runtime_class": os.getenv("SANDBOX_RUNTIME_CLASS", ""),
        "cpu_limit": os.getenv("SANDBOX_CPU_LIMIT", "2"),
        "memory_limit": os.getenv("SANDBOX_MEMORY_LIMIT", "4Gi"),
        "idle_timeout": int(os.getenv("SANDBOX_IDLE_TIMEOUT", "1800")),
        "max_lifetime": int(os.getenv("SANDBOX_MAX_LIFETIME", "86400")),
    }


def create_k8s_sandbox(sandbox_id: str | None = None):
    """Create or reconnect to a K8s sandbox.

    Creates a Sandbox CR via the Kubernetes API. The agent-controller
    watches the CR and creates the actual pod.

    Args:
        sandbox_id: Existing sandbox name to reconnect to.

    Returns:
        K8sPodBackend implementing SandboxBackendProtocol.
    """
    _load_k8s_config()
    core_api = client.CoreV1Api()
    custom_api = client.CustomObjectsApi()
    cfg = _get_config()
    namespace = cfg["namespace"]

    if sandbox_id:
        sb = custom_api.get_namespaced_custom_object(
            SANDBOX_API_GROUP, SANDBOX_API_VERSION, namespace, SANDBOX_PLURAL, sandbox_id
        )
        pod_name = sb.get("status", {}).get("podName", sandbox_id)
        return K8sPodBackend(
            pod_name=pod_name,
            namespace=namespace,
            core_api=core_api,
            sandbox_name=sandbox_id,
            custom_api=custom_api,
        )

    sandbox_name = f"sandbox-{int(time.time())}-{os.getpid()}"

    sandbox_body = {
        "apiVersion": f"{SANDBOX_API_GROUP}/{SANDBOX_API_VERSION}",
        "kind": "Sandbox",
        "metadata": {
            "name": sandbox_name,
            "namespace": namespace,
            "labels": {"mc.o8s.io/owner": "openswe"},
        },
        "spec": {
            "image": cfg["image"],
            "idleTimeoutSeconds": cfg["idle_timeout"],
            "maxLifetimeSeconds": cfg["max_lifetime"],
            "lastActivityAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    }

    if cfg["runtime_class"]:
        sandbox_body["spec"]["runtimeClassName"] = cfg["runtime_class"]

    if cfg["cpu_limit"] or cfg["memory_limit"]:
        sandbox_body["spec"]["resources"] = {}
        if cfg["cpu_limit"]:
            sandbox_body["spec"]["resources"]["cpu"] = cfg["cpu_limit"]
        if cfg["memory_limit"]:
            sandbox_body["spec"]["resources"]["memory"] = cfg["memory_limit"]

    custom_api.create_namespaced_custom_object(
        SANDBOX_API_GROUP, SANDBOX_API_VERSION, namespace, SANDBOX_PLURAL, sandbox_body
    )

    # Wait for sandbox to become Running
    timeout = 120
    poll_interval = 2
    elapsed = 0
    pod_name = sandbox_name

    while elapsed < timeout:
        sb = custom_api.get_namespaced_custom_object(
            SANDBOX_API_GROUP, SANDBOX_API_VERSION, namespace, SANDBOX_PLURAL, sandbox_name
        )
        phase = sb.get("status", {}).get("phase", "")
        if phase == "Running":
            pod_name = sb["status"].get("podName", sandbox_name)
            break
        if phase in ("Failed", "Terminated"):
            msg = sb.get("status", {}).get("message", "Unknown error")
            raise RuntimeError(f"Sandbox failed to start: {msg}")
        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        raise RuntimeError(f"Sandbox {sandbox_name} did not start within {timeout}s")

    backend = K8sPodBackend(
        pod_name=pod_name,
        namespace=namespace,
        core_api=core_api,
        sandbox_name=sandbox_name,
        custom_api=custom_api,
    )

    _update_thread_sandbox_metadata(sandbox_name)

    return backend


def _update_thread_sandbox_metadata(sandbox_id: str) -> None:
    """Update thread metadata with sandbox_id (best-effort)."""
    try:
        import asyncio

        from langgraph.config import get_config
        from langgraph_sdk import get_client

        config_data = get_config()
        thread_id = config_data.get("configurable", {}).get("thread_id")
        if not thread_id:
            return
        lc = get_client()

        async def _update():
            await lc.threads.update(thread_id=thread_id, metadata={"sandbox_id": sandbox_id})

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_update())
        else:
            loop.create_task(_update())
    except Exception:
        pass


class K8sPodBackend(BaseSandbox):
    """Kubernetes pod sandbox backend.

    Executes commands via the Kubernetes exec API in a pod
    managed by the agent-controller's Sandbox CRD.
    """

    def __init__(
        self,
        pod_name: str,
        namespace: str,
        core_api: client.CoreV1Api,
        sandbox_name: str,
        custom_api: client.CustomObjectsApi,
    ):
        self._pod_name = pod_name
        self._namespace = namespace
        self._core_api = core_api
        self._sandbox_name = sandbox_name
        self._custom_api = custom_api
        self._default_timeout = 300

    @property
    def id(self) -> str:
        return self._sandbox_name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a command in the sandbox pod via K8s exec API."""
        effective_timeout = timeout if timeout is not None else self._default_timeout

        try:
            resp = k8s_stream(
                self._core_api.connect_get_namespaced_pod_exec,
                self._pod_name,
                self._namespace,
                command=["/bin/sh", "-c", command],
                container="sandbox",
                stderr=True,
                stdout=True,
                stdin=False,
                tty=False,
                _preload_content=False,
            )

            resp.run_forever(timeout=effective_timeout)

            stdout = resp.read_stdout() or ""
            stderr = resp.read_stderr() or ""

            output = stdout
            if stderr:
                output += "\n" + stderr if output else stderr

            # Parse exit code from status channel (channel 3)
            exit_code = 0
            err = resp.read_channel(3)
            if err:
                try:
                    status = json.loads(err)
                    if status.get("status") == "Failure":
                        causes = status.get("details", {}).get("causes", [])
                        for cause in causes:
                            if cause.get("reason") == "ExitCode":
                                exit_code = int(cause.get("message", "1"))
                                break
                        else:
                            exit_code = 1
                except (json.JSONDecodeError, ValueError):
                    exit_code = 1

            resp.close()
        except Exception as e:
            return ExecuteResponse(
                output=f"K8s exec failed: {e}",
                exit_code=1,
                truncated=False,
            )

        self._touch_activity()

        return ExecuteResponse(output=output, exit_code=exit_code, truncated=False)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write file via stdin streaming to avoid ARG_MAX limits."""
        try:
            self.execute(f"mkdir -p $(dirname '{file_path}')")

            resp = k8s_stream(
                self._core_api.connect_get_namespaced_pod_exec,
                self._pod_name,
                self._namespace,
                command=["/bin/sh", "-c", f"cat > '{file_path}'"],
                container="sandbox",
                stderr=True,
                stdout=True,
                stdin=True,
                tty=False,
                _preload_content=False,
            )

            resp.write_stdin(content)
            resp.close()

            return WriteResult(path=file_path, files_update=None)
        except Exception as e:
            return WriteResult(error=f"Failed to write file '{file_path}': {e}")

    def _touch_activity(self) -> None:
        """Update spec.lastActivityAt on the Sandbox CR (best-effort)."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        patch = {"spec": {"lastActivityAt": now}}
        for attempt in range(3):
            try:
                self._custom_api.patch_namespaced_custom_object(
                    SANDBOX_API_GROUP,
                    SANDBOX_API_VERSION,
                    self._namespace,
                    SANDBOX_PLURAL,
                    self._sandbox_name,
                    patch,
                )
                return
            except Exception:
                if attempt < 2:
                    time.sleep(0.5)
