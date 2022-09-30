#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[8]:


init_jupyter()
from pydantic import BaseModel
from functools import singledispatchmethod
from abc import ABCMeta, abstractmethod, ABC
from pydantic.main import ModelMetaclass

class PydanticWithABCMeta(ModelMetaclass, ABCMeta):
    pass

class Payment(BaseModel):
    ...
class ScoringData(BaseModel, ABC):
    @classmethod
    def _build_from_payment(cls, payment: Payment):
        return cls._build_from_payment_impl(payment)

    @classmethod
    @abstractmethod
    def _build_from_payment_impl(cls, payment: Payment):
        ...

class TestData(ScoringData):
    a:int 
    @classmethod
    def _build_from_payment_impl(cls, payment: Payment):
        return cls(a=123)
    


# In[12]:


from pydantic import BaseModel
class ABTestEntry(BaseModel):
    env_key: str  # environ key is in percents
    tag_in: str = None
    tag_out: str = None
    default_prob: float = None
    prob: float = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.env_key not in os.environ:
            assert self.default_prob is not None, f"Bad init for {self.env_key}"
            self.prob = self.default_prob
        else:
            self.prob = float(os.environ[self.env_key]) / 100

    def accept(self, idx: str, add_tag_func):
        if self.prob == 0:
            return False
        mod = int(10003)
        h = (
            int.from_bytes(idx.encode(), byteorder="big")
            ^ int.from_bytes(self.env_key.encode(), byteorder="big")
        ) % mod

        enroll = h < self.prob * mod
        tag = self.tag_in if enroll else self.tag_out
        if tag and add_tag_func is not None:
            add_tag_func(tag)
        return enroll

    def __hash__(self):
        return self.env_key.__hash__()



# In[16]:


import os

a = ABTestEntry(env_key='123', default_prob=0.1)


# In[22]:


from enum import Enum
class X(Enum):
    a = 1
    b = 2
list(X)


# In[ ]:





# In[21]:


setattr(a, 'accept', 123)


# In[9]:


TestData._build_from_payment(Payment())


# In[ ]:


from pydantic.main import Mo


# In[ ]:





# In[4]:


class SpatialVectorBase(BaseModel):
    t: np.ndarray
    r: np.ndarray
    class Config:
        arbitrary_types_allowed = True

class SpatialVector(SpatialVectorBase):
    pass
class SpatialForce(SpatialVectorBase):
    pass


# In[9]:


from typing import Any
from abc import ABC, abstractmethod
subclass_registry = {}

class ModifiedBaseModel(BaseModel):
    def __init_subclass__(cls, is_abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)  # type: ignore
        if not is_abstract:
            subclass_registry[cls.__name__] = cls


class Pet(ModifiedBaseModel, ABC, is_abstract=True):
    name: str
    age: int = 2

    class Config:
        extra = "allow"

    @abstractmethod
    def get_relative_age(self) -> int:
        raise NotImplementedError


class Dog(Pet):

    name: str = "Fido"
    food: str = "kibble"

    def get_relative_age(self) -> int:
        return self.age * 7


class Cat(Pet):

    name: str = "Fluffy"
    collar_color: str = "Red"

    def get_relative_age(self) -> int:
        return self.age * 4


class Owner(ModifiedBaseModel):
    pets: list[Pet]

Owner(pets=[Dog(name='Fido', age=2, food='kibble'), Cat(name='Fluffy', age=2, collar_color='Red')])

