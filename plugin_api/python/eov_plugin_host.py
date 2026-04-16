from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import queue
from typing import Iterator

import grpc
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


_PACKAGE = "eov.extension"
_SERVICE = f"/{_PACKAGE}.ExtensionHost"
_DESCRIPTOR_PATH = Path(__file__).with_name("eov_extension.desc")
_MAX_GRPC_MESSAGE_LENGTH = 64 * 1024 * 1024


def _load_pool() -> descriptor_pool.DescriptorPool:
    file_set = descriptor_pb2.FileDescriptorSet()
    file_set.ParseFromString(_DESCRIPTOR_PATH.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for file_proto in file_set.file:
        pool.Add(file_proto)
    return pool


_POOL = _load_pool()


def _message_class(name: str):
    descriptor = _POOL.FindMessageTypeByName(f"{_PACKAGE}.{name}")
    return message_factory.GetMessageClass(descriptor)


def _enum_number(enum_name: str, value_name: str) -> int:
    descriptor = _POOL.FindEnumTypeByName(f"{_PACKAGE}.{enum_name}")
    return descriptor.values_by_name[value_name].number


@dataclass(frozen=True)
class ToolbarAction:
    plugin_handle: str
    plugin_id: str
    button_id: str
    action_id: str


@dataclass(frozen=True)
class HudToolbarAction:
    plugin_handle: str
    plugin_id: str
    button_id: str
    action_id: str
    viewport: object


@dataclass(frozen=True)
class CpuFilterFrame:
    width: int
    height: int
    rgba_data: bytes


class CpuFilterStream:
    def __init__(self, call, request_cls, filter_id: str):
        self._request_cls = request_cls
        self._filter_id = filter_id
        self._requests: queue.Queue = queue.Queue()
        self._responses = call(self._request_iterator())

    def _request_iterator(self):
        yield self._request_cls(filter_id=self._filter_id)
        while True:
            message = self._requests.get()
            if message is None:
                break
            yield message

    def __iter__(self) -> Iterator[CpuFilterFrame]:
        for response in self._responses:
            yield CpuFilterFrame(
                width=response.width,
                height=response.height,
                rgba_data=response.rgba_data,
            )

    def send_processed_frame(self, width: int, height: int, rgba_data: bytes) -> None:
        self._requests.put(
            self._request_cls(
                filter_id=self._filter_id,
                width=width,
                height=height,
                rgba_data=rgba_data,
            )
        )

    def close(self) -> None:
        self._requests.put(None)


class RemotePluginSession:
    def __init__(self, client: "ExtensionHostClient", response):
        self._client = client
        self.plugin_handle = response.plugin_handle
        self.plugin_id = response.plugin_id
        self.initial_snapshot = response.host_snapshot
        self._closed = False

    def unregister(self) -> None:
        if self._closed:
            return
        request = self._client.messages.UnregisterPluginRequest(
            plugin_handle=self.plugin_handle,
        )
        self._client.calls.unregister_plugin(request)
        self._closed = True

    def log_message(self, level: str, message: str) -> None:
        enum_name = {
            "trace": "LOG_LEVEL_TRACE",
            "debug": "LOG_LEVEL_DEBUG",
            "info": "LOG_LEVEL_INFO",
            "warn": "LOG_LEVEL_WARN",
            "error": "LOG_LEVEL_ERROR",
        }[level.lower()]
        request = self._client.messages.LogMessageRequest(
            plugin_handle=self.plugin_handle,
            level=_enum_number("LogLevel", enum_name),
            message=message,
        )
        self._client.calls.log_message(request)

    def register_toolbar_button(
        self,
        button_id: str,
        tooltip: str,
        action_id: str,
        icon_svg: str = "",
    ) -> None:
        request = self._client.messages.RegisterToolbarButtonRequest(
            plugin_handle=self.plugin_handle,
            button_id=button_id,
            tooltip=tooltip,
            action_id=action_id,
            icon_svg=icon_svg,
        )
        self._client.calls.register_toolbar_button(request)

    def set_toolbar_button_active(self, button_id: str, active: bool) -> None:
        request = self._client.messages.SetToolbarButtonActiveRequest(
            plugin_handle=self.plugin_handle,
            button_id=button_id,
            active=active,
        )
        self._client.calls.set_toolbar_button_active(request)

    def register_hud_toolbar_button(
        self,
        button_id: str,
        tooltip: str,
        icon_svg: str,
        action_id: str,
    ) -> None:
        request = self._client.messages.RegisterHudToolbarButtonRequest(
            plugin_handle=self.plugin_handle,
            button_id=button_id,
            tooltip=tooltip,
            icon_svg=icon_svg,
            action_id=action_id,
        )
        self._client.calls.register_hud_toolbar_button(request)

    def set_hud_toolbar_button_active(self, button_id: str, active: bool) -> None:
        request = self._client.messages.SetHudToolbarButtonActiveRequest(
            plugin_handle=self.plugin_handle,
            button_id=button_id,
            active=active,
        )
        self._client.calls.set_hud_toolbar_button_active(request)

    def toolbar_action_stream(self):
        request = self._client.messages.ToolbarActionStreamRequest(
            plugin_handle=self.plugin_handle,
        )
        for response in self._client.calls.toolbar_action_stream(request):
            yield ToolbarAction(
                plugin_handle=response.plugin_handle,
                plugin_id=response.plugin_id,
                button_id=response.button_id,
                action_id=response.action_id,
            )

    def hud_toolbar_action_stream(self):
        request = self._client.messages.HudToolbarActionStreamRequest(
            plugin_handle=self.plugin_handle,
        )
        for response in self._client.calls.hud_toolbar_action_stream(request):
            yield HudToolbarAction(
                plugin_handle=response.plugin_handle,
                plugin_id=response.plugin_id,
                button_id=response.button_id,
                action_id=response.action_id,
                viewport=response.viewport,
            )

    def register_filter(self, name: str, supports_cpu: bool, supports_gpu: bool) -> str:
        request = self._client.messages.RegisterFilterRequest(
            plugin_handle=self.plugin_handle,
            name=name,
            supports_cpu=supports_cpu,
            supports_gpu=supports_gpu,
        )
        response = self._client.calls.register_filter(request)
        return response.filter_id

    def set_filter_enabled(self, filter_id: str, enabled: bool) -> None:
        request = self._client.messages.SetFilterEnabledRequest(
            plugin_handle=self.plugin_handle,
            filter_id=filter_id,
            enabled=enabled,
        )
        self._client.calls.set_filter_enabled(request)

    def open_cpu_filter_stream(self, filter_id: str) -> CpuFilterStream:
        return CpuFilterStream(
            self._client.calls.apply_filter_cpu_stream,
            self._client.messages.ApplyFilterCpuRequest,
            filter_id,
        )

    def get_host_snapshot(self):
        return self._client.get_host_snapshot()

    def read_region(
        self,
        file_id: int,
        level: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bytes:
        return self._client.read_region(file_id, level, x, y, width, height)

    def open_file(self, path: str) -> None:
        self._client.open_file(path)

    def set_active_viewport(self, center_x: float, center_y: float, zoom: float) -> None:
        self._client.set_active_viewport(center_x, center_y, zoom)

    def fit_active_viewport(self) -> None:
        self._client.fit_active_viewport()

    def frame_active_rect(self, x: float, y: float, width: float, height: float) -> None:
        self._client.frame_active_rect(x, y, width, height)


class _MessageNamespace:
    def __getattr__(self, name: str):
        return _message_class(name)


class _CallNamespace:
    def __init__(self, channel: grpc.Channel):
        self._channel = channel

    def _unary_unary(self, method: str, request_name: str, response_name: str):
        request_cls = _message_class(request_name)
        response_cls = _message_class(response_name)
        return self._channel.unary_unary(
            f"{_SERVICE}/{method}",
            request_serializer=request_cls.SerializeToString,
            response_deserializer=response_cls.FromString,
        )

    def _unary_stream(self, method: str, request_name: str, response_name: str):
        request_cls = _message_class(request_name)
        response_cls = _message_class(response_name)
        return self._channel.unary_stream(
            f"{_SERVICE}/{method}",
            request_serializer=request_cls.SerializeToString,
            response_deserializer=response_cls.FromString,
        )

    def _stream_stream(self, method: str, request_name: str, response_name: str):
        request_cls = _message_class(request_name)
        response_cls = _message_class(response_name)
        return self._channel.stream_stream(
            f"{_SERVICE}/{method}",
            request_serializer=request_cls.SerializeToString,
            response_deserializer=response_cls.FromString,
        )

    @property
    def register_plugin(self):
        return self._unary_unary("RegisterPlugin", "RegisterPluginRequest", "RegisterPluginResponse")

    @property
    def unregister_plugin(self):
        return self._unary_unary("UnregisterPlugin", "UnregisterPluginRequest", "HostCommandResponse")

    @property
    def get_host_snapshot(self):
        return self._unary_unary("GetHostSnapshot", "Empty", "HostSnapshot")

    @property
    def read_region(self):
        return self._unary_unary("ReadRegion", "ReadRegionRequest", "ReadRegionResponse")

    @property
    def open_file(self):
        return self._unary_unary("OpenFile", "OpenFileRequest", "HostCommandResponse")

    @property
    def set_active_viewport(self):
        return self._unary_unary("SetActiveViewport", "SetActiveViewportRequest", "HostCommandResponse")

    @property
    def fit_active_viewport(self):
        return self._unary_unary("FitActiveViewport", "Empty", "HostCommandResponse")

    @property
    def frame_active_rect(self):
        return self._unary_unary("FrameActiveRect", "FrameActiveRectRequest", "HostCommandResponse")

    @property
    def log_message(self):
        return self._unary_unary("LogMessage", "LogMessageRequest", "HostCommandResponse")

    @property
    def register_toolbar_button(self):
        return self._unary_unary("RegisterToolbarButton", "RegisterToolbarButtonRequest", "HostCommandResponse")

    @property
    def toolbar_action_stream(self):
        return self._unary_stream("ToolbarActionStream", "ToolbarActionStreamRequest", "ToolbarActionRequest")

    @property
    def set_toolbar_button_active(self):
        return self._unary_unary("SetToolbarButtonActive", "SetToolbarButtonActiveRequest", "HostCommandResponse")

    @property
    def register_hud_toolbar_button(self):
        return self._unary_unary("RegisterHudToolbarButton", "RegisterHudToolbarButtonRequest", "HostCommandResponse")

    @property
    def set_hud_toolbar_button_active(self):
        return self._unary_unary("SetHudToolbarButtonActive", "SetHudToolbarButtonActiveRequest", "HostCommandResponse")

    @property
    def hud_toolbar_action_stream(self):
        return self._unary_stream("HudToolbarActionStream", "HudToolbarActionStreamRequest", "HudToolbarActionRequest")

    @property
    def register_filter(self):
        return self._unary_unary("RegisterFilter", "RegisterFilterRequest", "RegisterFilterResponse")

    @property
    def set_filter_enabled(self):
        return self._unary_unary("SetFilterEnabled", "SetFilterEnabledRequest", "SetFilterEnabledResponse")

    @property
    def apply_filter_cpu_stream(self):
        return self._stream_stream("ApplyFilterCpuStream", "ApplyFilterCpuRequest", "ApplyFilterCpuResponse")


class ExtensionHostClient:
    def __init__(self, target: str):
        self.channel = grpc.insecure_channel(
            target.replace("grpc://", ""),
            options=[
                ("grpc.max_send_message_length", _MAX_GRPC_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", _MAX_GRPC_MESSAGE_LENGTH),
            ],
        )
        self.messages = _MessageNamespace()
        self.calls = _CallNamespace(self.channel)

    @classmethod
    def from_env(cls, env_var: str = "EOV_EXTENSION_HOST") -> "ExtensionHostClient":
        target = os.environ.get(env_var)
        if not target:
            raise RuntimeError(f"{env_var} is not set")
        return cls(target)

    def get_host_snapshot(self):
        return self.calls.get_host_snapshot(self.messages.Empty())

    def read_region(
        self,
        file_id: int,
        level: int,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bytes:
        request = self.messages.ReadRegionRequest(
            file_id=file_id,
            level=level,
            x=x,
            y=y,
            width=width,
            height=height,
        )
        response = self.calls.read_region(request)
        return response.rgba_data

    def open_file(self, path: str) -> None:
        self.calls.open_file(self.messages.OpenFileRequest(path=path))

    def set_active_viewport(self, center_x: float, center_y: float, zoom: float) -> None:
        self.calls.set_active_viewport(
            self.messages.SetActiveViewportRequest(
                center_x=center_x,
                center_y=center_y,
                zoom=zoom,
            )
        )

    def fit_active_viewport(self) -> None:
        self.calls.fit_active_viewport(self.messages.Empty())

    def frame_active_rect(self, x: float, y: float, width: float, height: float) -> None:
        self.calls.frame_active_rect(
            self.messages.FrameActiveRectRequest(x=x, y=y, width=width, height=height)
        )

    def register_plugin(
        self,
        plugin_id: str,
        display_name: str | None = None,
        version: str = "",
        language: str = "python",
    ) -> RemotePluginSession:
        request = self.messages.RegisterPluginRequest(
            plugin_id=plugin_id,
            display_name=display_name or plugin_id,
            version=version,
            language=language,
        )
        response = self.calls.register_plugin(request)
        return RemotePluginSession(self, response)

    def close(self) -> None:
        self.channel.close()

    def __enter__(self) -> "ExtensionHostClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "CpuFilterFrame",
    "CpuFilterStream",
    "ExtensionHostClient",
    "HudToolbarAction",
    "RemotePluginSession",
    "ToolbarAction",
]