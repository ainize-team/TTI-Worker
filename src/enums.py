from enum import Enum


class StrEnum(str, Enum):
    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value


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
    PROD: str = "prod"


class ErrorStatusEnum(IntEnum):
    UNPROCESSABLE_ENTITY = 422
    INTERNAL_SERVER_ERROR = 500


class ModelTypeEnums(StrEnum):
    STABLE_DIFFUSION_V1: str = "STABLE_DIFFUSION_V1"
    STABLE_DIFFUSION_V2: str = "STABLE_DIFFUSION_V2"
    LDM_TEXT_TO_IMAGE: str = "LDM_TEXT_TO_IMAGE"
