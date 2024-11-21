"""
'baba is you' is a valid sentence or rule comprised of 3 consecutive text blocks in the 'world':
- 'baba' (subject): (noun) text block pointing to the image block of the same name
- 'is' (verb): (verb) text block pointing to the action to be performed
- 'you' (object): () is the object point to where my playable character

rules:
- 'baba is push': each IBaba object should be pushed out of the square it is in if an object moves onto it
- 'baba is baba': each IBaba object should stay as that object
- 'baba is wall': each IBaba object should be replaced with a IWall object
- 'baba is you':  each IBaba object should be controlled by the user
- 'baba is win':  each IBaba object should be the winning location
- 'baba makes wall': each IBaba object should produce a new IWall object in place
- 'baba is open': each IBaba object can remove a shut object, removing itself in the process

- rules:
  - subject verb object, where
    - subject:
      - noun [conjunction noun [...]]
    - object:
      - noun/adjective [conjunction noun/adjective [...]]
  - nouns can be negated
  - adjectives can be negated

block (sub)types:
- text
  - noun (baba, wall, robot, me, text, empty, ...)
  - verb (is, has, makes, ...)
  - adjective (push, you, pull, win, shift, move, defeat, tele, ...)
  - conjunction (and, on)
  - prefix (not)
    
"""
from enum import IntEnum
from pydantic import BaseModel
from typing import Self

class Block(BaseModel):
    pass

# *** words ************
# blocks that only ever appear as a single text word
# **********************

class Word(Block):
    pass


class Conjunction(Word):
    pass

class And(Conjunction):
    pass

# TODO: consider this a conditional conjunction?
class On(Conjunction):
    pass

CONJUNCTION_CLASSES = (And, On)


class Prefix(Word):
    pass

class Not(Prefix):
    pass

PREFIX_CLASSES = (Not,)


class Verb(Word):
    pass

class Is(Verb):
    pass

class Has(Verb):
    pass

class Makes(Verb):
    pass

VERB_CLASSES = (Is, Has, Makes)


class Adjective(Word):
    pass

class You(Adjective):
    pass

class Win(Adjective):
    pass

class Shift(Adjective):
    pass

class Stop(Adjective):
    pass

class Push(Adjective):
    pass

class Pull(Adjective):
    pass

ADJECTIVE_CLASSES = (You, Win, Shift, Stop, Push, Pull)


class Noun(Word):
    pass
    # def image_cls(self) -> type[Block]:
    #     raise NotImplementedError()

class Baba(Noun):
    pass

class Flag(Noun):
    pass

class Text(Noun):
    pass

class Me(Noun):
    pass

class Robot(Noun):
    pass

class Empty(Noun):
    pass


NOUN_CLASSES = (Baba, Flag, Text, Me, Robot, Empty)


# *** images ***********
# blocks that only ever appear as images
# **********************

class Image(Block):
    pass

class IBaba(Image):
    pass

class IFlag(Image):
    pass

class IMe(Image):
    pass

class IRobot(Image):
    pass

class IEmpty(Image):
    pass

IMAGE_CLASSES = (IBaba, IFlag, IMe, IRobot, IEmpty)

        
class Sentence(BaseModel):
    """
    an arbitrary sequence of word blocks, with methods to pull out valid sentence structures
    
    valid rule structure: subject verb object, where
    - subject:
      - simple: [prefix] noun
        - e.g.:
          - baba
          - not baba
      - complex: simple [conjunction simple [conjuction simple]]
        - e.g.:
          - baba
          - not baba
          - baba and robot
          - baba on robot
          - not baba and not robot
    - object:
      - simple: [prefix] noun | [prefix] adjective
        - e.g.:
          - baba
          - not baba
          - shift
          - not shift
      - complex: simple [conjunction simple [conjuction simple]]
        - e.g.:
          - baba
          - not baba
          - baba and robot
          - baba on robot
          - not baba and not robot
          - baba and shift
          - not baba and not shift
          - move and shift
          - not move and not shift
          
    make into binary tree?
    - subject verb object
    - note: subject is more restrictive than object, i.e. any subject can be an object
    - parse first come first serve, meaning, leftmost valid structure is first, remove it from sequence, repeat
    
    """
    words: list[Word]
    
    @property
    def num_words(self) -> int:
        return len(self.words)
    
    def to_leftmost_rule(self, max_depth: int = 10) -> Self | None:
        verb_idxs = [i for i, word in enumerate(self.words) if isinstance(word, Verb)]
        start_idxs = [0] + [i+1 for i in verb_idxs]
        verb_idxs.append(self.num_words)
        end_idxs = verb_idxs[1:]

        for start_idx, verb_idx, end_idx in zip(start_idxs, verb_idxs, end_idxs):
            verb = self.words[verb_idx]
            # print(f'{left_idx=}, {verb_idx=}, {end_idx=}, {self.num_words=}')
            for left_idx in range(start_idx, verb_idx):
                left_words = self.words[left_idx:verb_idx]
                # print(f'{left_idx=}, {left_words=}')
                for right_idx in range(end_idx, verb_idx, -1):
                    right_words = self.words[verb_idx+1:right_idx]
                    # print(f'{right_idx=}, {right_words=}')
                    if Sentence(words=left_words).is_subject(max_depth=max_depth) and Sentence(words=right_words).is_object(max_depth=max_depth):
                        return Sentence(words=left_words + [verb] + right_words)
        else:
            return None
        
    def is_rule(self, max_depth: int = 10) -> bool:
        # rule means: subject verb object
        for verb_idx, word in enumerate(self.words):
            if isinstance(word, Verb):
                break
        else:
            return False
        left_words = self.words[:verb_idx]
        right_words = self.words[verb_idx+1:]
        # print(f'{verb_idx=}, {left_words=}, {right_words=}')
        return Sentence(words=left_words).is_subject(max_depth=max_depth) and Sentence(words=right_words).is_object(max_depth=max_depth)
    
    def is_subject(self, max_depth: int = 10):
        # subject means: noun [conj noun [conj noun...]]
        return self.is_complex(restrict_to_noun_only=True, max_depth=max_depth)
    
    def is_object(self, max_depth: int = 10):
        # object means: noun|adj [conj noun|adj [conj noun|adj...]]
        return self.is_complex(restrict_to_noun_only=False, max_depth=max_depth)
    
    def is_complex(self, restrict_to_noun_only: bool, max_depth: int = 10):
        # complex means: simple [conj simple [conj simple...]]
        lst = []
        n_found = 0
        for word in self.words:
            if isinstance(word, Conjunction):
                if Sentence(words=lst).is_simple(restrict_to_noun_only=restrict_to_noun_only) and n_found < max_depth:
                    n_found += 1
                    lst = []
                else:
                    return False
            else:
                lst.append(word)
        # last bit after last conjunction should be simple
        return Sentence(words=lst).is_simple(restrict_to_noun_only=restrict_to_noun_only)
    
    def is_simple(self, restrict_to_noun_only: bool) -> bool:
        # simple means: [prefix] noun/adj
        Cls = Noun if restrict_to_noun_only else (Noun, Adjective)
        if self.num_words < 1 or self.num_words > 2:
            return False
        else:
            last_word = self.words[-1]
            # print(f'{last_word=}, {Cls=}, {(not isinstance(last_word, Cls))=}, {self.num_words=}')
            if not isinstance(last_word, Cls):
                return False
            elif (self.num_words == 2) and not isinstance(self.words[0], Prefix):
                return False
            return True


class Location(BaseModel):
    x: int
    y: int

    def __add__(self, other: "Location") -> "Location":
        o = other.as_loc() if isinstance(other, Direction) else other
        return Location(
            x=self.x + o.x,
            y=self.y + o.y,
        )

    # # @overload
    # def __add__(self, other: "Direction") -> "Location":
    #     return self + other.as_loc()
        
class Direction(IntEnum):
    # direction, tied with how it'd add to a location
    LEFT  = -2
    DOWN  = -1
    STAY  = 0
    UP    = 1
    RIGHT = 2
    
    def as_loc(self, rel_to: Location = Location(x=0, y=0)) -> Location:
        # return location relative to origin (0, 0)
        return rel_to + Location(
            x=(1 if self == self.RIGHT else (-1 if self == self.LEFT else 0)),
            y=(1 if self == self.UP else (-1 if self == self.DOWN else 0)),
        )

if __name__ == "__main__":
    s = Sentence(words=[You(), Is(), Baba(), And(), Robot(), Is(), Push(), And(), You(), Is(), Baba()])
    print(f'{s=}')
    r = s.to_leftmost_rule()
    print(f'{r=}')
    
    d = Direction.DOWN
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')
    d = Direction.UP
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')
    d = Direction.LEFT
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')
    d = Direction.RIGHT
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')
    d = Direction.STAY
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')

    loc = Location(x=0, y=0)
    print(f'{loc=}')
    new_loc = loc + Direction.DOWN
    print(f'{new_loc=}')
    # assert (new_loc.x == loc.x) and (new_loc.y == loc.y - 1)