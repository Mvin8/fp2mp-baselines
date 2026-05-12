import uuid

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, model_validator


def get_id(length: int = 6) -> str:
    return uuid.uuid4().hex[:length]


class BaseNote(BaseModel):
    """Запись для добавления на доску."""

    content: str = Field(description="Содержимое записи")


class BaseRole(BaseModel):
    """Роль агента для работы с общей доской."""

    name: str = Field(description="Название роли")
    description: str = Field(description="Краткое описание экспертизы роли")


class Note(BaseNote):
    id: str = Field(default="", description="ID записи")
    author_id: str = Field(description="ID автора записи")
    author_role: str = Field(default="", description="Роль автора записи")

    @model_validator(mode="after")
    def _set_id(self):
        if not self.id:
            self.id = get_id()
        return self


class Board(BaseModel):
    notes: list[Note] = Field(default_factory=list, description="Записи на доске")

    def add_note(self, base_note: BaseNote, author_id: str, author_role: str = "") -> str:
        note = Note(author_id=author_id, author_role=author_role, **base_note.model_dump())
        self.notes.append(note)
        return note.id

    def remove_notes(self, notes_ids: list[str]) -> None:
        self.notes = [note for note in self.notes if note.id not in notes_ids]

    def to_messages(self) -> list[HumanMessage]:
        if not self.notes:
            return [HumanMessage(content="На общей доске пока нет сообщений.")]

        messages: list[HumanMessage] = []
        for index, note in enumerate(self.notes, start=1):
            messages.append(
                HumanMessage(
                    content=(
                        f"Сообщение на общей доске #{index}\n"
                        f"ID: {note.id}\n"
                        f"Автор: {note.author_role or 'неизвестная роль'} ({note.author_id})\n"
                        f"Содержание:\n{note.content}"
                    )
                )
            )
        return messages

    def to_str(self) -> str:
        return "Содержимое доски:\n" + str([note.model_dump() for note in self.notes])


__all__ = ["BaseNote", "BaseRole", "Board", "Note", "get_id"]
