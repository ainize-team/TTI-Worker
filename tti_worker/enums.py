from enum import Enum


class StrEnum(str, Enum):
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class IntEnum(int, Enum):
    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)


class ResponseStatusEnum(StrEnum):
    PENDING: str = "pending"
    ASSIGNED: str = "assigned"
    COMPLETED: str = "completed"
    ERROR: str = "error"


class EnvEnum(StrEnum):
    DEV: str = "dev"
    STAGGING: str = "stagging"
    PROD: str = "prod"


class ErrorStatusEnum(IntEnum):
    UNPROCESSABLE_ENTITY = 422
    INTERNAL_SERVER_ERROR = 500
