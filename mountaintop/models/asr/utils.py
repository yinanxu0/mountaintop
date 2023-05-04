
from enum import Enum, IntEnum


class EncoderMode(Enum):
    ### train = Offline + DynamicChunk
    ### decode = Offline + StaticChunk
    Offline = 1
    DynamicChunk = 2
    StaticChunk = 3
    Stream = 4
    NotValid = 9999

    def is_valid(self):
        return self != self.NotValid
    
    @classmethod
    def to_enum(cls, name: str) :
        if name.lower() == "offline":
            return cls.Offline
        elif name.lower() in ["dynamicchunk", "dynamic_chunk"]:
            return cls.DynamicChunk
        elif name.lower() in ["staticchunk", "static_chunk"]:
            return cls.StaticChunk
        elif name.lower() == "stream":
            return cls.Stream
        else:
            return cls.NotValid