"""Synchronous, explicitly counted FIPA-inspired message delivery within each Mesa tick."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import mesa


@dataclass(frozen=True)
class Message:
    performative: str
    sender: str
    receiver: str
    content: dict[str, Any]
    conversation: str
    tick: int


@dataclass(frozen=True)
class Reply:
    performative: str
    content: dict[str, Any]


class MessageBus:
    def __init__(self, model):
        self.model = model
        self.endpoints = {}
        self.counts = Counter()
        self.by_sender = Counter()
        self.trace = []

    def register(self, name: str, agent):
        self.endpoints[name] = agent

    def _record(self, message: Message):
        self.counts[message.performative] += 1
        self.by_sender[message.sender] += 1
        if self.model.record_trace:
            # Contents may contain immutable routing objects; traces retain protocol metadata.
            self.trace.append(
                {
                    "tick": message.tick,
                    "performative": message.performative,
                    "sender": message.sender,
                    "receiver": message.receiver,
                    "conversation": message.conversation,
                }
            )

    def send(self, performative, sender, receiver, content, conversation="system", tick=None):
        timestamp = self.model.tick if tick is None else tick
        message = Message(performative, sender, receiver, content, conversation, timestamp)
        self._record(message)
        response = self.endpoints[receiver].receive(message)
        if response is None:
            return None
        self._record(
            Message(
                response.performative,
                receiver,
                sender,
                response.content,
                conversation,
                timestamp,
            )
        )
        return response.content


class ProtocolAgent(mesa.Agent):
    def __init__(self, model, name):
        super().__init__(model)
        self.name = name
        model.bus.register(name, self)

    def send(self, performative, receiver, content, conversation="system", tick=None):
        return self.model.bus.send(performative, self.name, receiver, content, conversation, tick)

    def receive(self, message: Message) -> Reply | None:
        raise NotImplementedError
