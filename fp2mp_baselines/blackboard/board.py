import uuid

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, model_validator


def get_id(length: int = 6) -> str:
    return uuid.uuid4().hex[:length]


class BaseNote(BaseModel):
    """A note to add to the blackboard."""

    content: str = Field(description="Note content")


class BaseRole(BaseModel):
    """An agent role for working with the shared blackboard."""

    name: str = Field(description="Role name")
    description: str = Field(description="Short description of the role's expertise")


class Note(BaseNote):
    id: str = Field(default="", description="Note ID")
    author_id: str = Field(description="Author ID")
    author_role: str = Field(default="", description="Author role")

    @model_validator(mode="after")
    def _set_id(self):
        if not self.id:
            self.id = get_id()
        return self


class Board(BaseModel):
    notes: list[Note] = Field(default_factory=list, description="Blackboard notes")

    def add_note(self, base_note: BaseNote, author_id: str, author_role: str = "") -> str:
        note = Note(author_id=author_id, author_role=author_role, **base_note.model_dump())
        self.notes.append(note)
        return note.id

    def remove_notes(self, notes_ids: list[str]) -> None:
        self.notes = [note for note in self.notes if note.id not in notes_ids]

    def to_messages(self) -> list[HumanMessage]:
        if not self.notes:
            return [HumanMessage(content="There are no messages on the shared blackboard yet.")]

        messages: list[HumanMessage] = []
        for index, note in enumerate(self.notes, start=1):
            messages.append(
                HumanMessage(
                    content=(
                        f"Shared blackboard message #{index}\n"
                        f"ID: {note.id}\n"
                        f"Author: {note.author_role or 'unknown role'} ({note.author_id})\n"
                        f"Content:\n{note.content}"
                    )
                )
            )
        return messages

    def to_str(self) -> str:
        return "Blackboard contents:\n" + str([note.model_dump() for note in self.notes])


__all__ = ["BaseNote", "BaseRole", "Board", "Note", "get_id"]
