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


class SchedulerType(StrEnum):
    DDIM: str = "ddim"  # DDIMScheduler
    PNDM: str = "pndm"  # PNDMScheduler
    EULER_DISCRETE = "euler_discrete"  # EulerDiscreteScheduler
    EULER_ANCESTRAL_DISCRETE = "euler_ancestral_discrete"  # EulerAncestralDiscreteScheduler
    HEUN_DISCRETE = "heun_discrete"  # HeunDiscreteScheduler
    K_DPM_2_DISCRETE = "k_dpm_2_discrete"  # KDPM2DiscreteScheduler
    K_DPM_2_ANCESTRAL_DISCRETE = "k_dpm_2_ancestral_discrete"  # KDPM2AncestralDiscreteScheduler
    LMS_DISCRETE = "lms_discrete"  # LMSDiscreteScheduler
