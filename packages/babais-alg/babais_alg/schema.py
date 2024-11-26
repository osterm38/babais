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
from itertools import product, chain
# import numpy as np
# import numpy.typing as npt
from pydantic import BaseModel, Field, NonNegativeInt, conlist
from typing import Self, Iterator, Annotated


class Location(BaseModel):
    x: int
    y: int

    def __add__(self, other: "Location") -> "Location":
        o = other.as_loc() if isinstance(other, Direction) else other
        return Location(
            x=self.x + o.x,
            y=self.y + o.y,
        )


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

class Block(BaseModel):
    pass
    # facing: Direction = Direction.DOWN

# *** words ************
# blocks that only ever appear as a single text word
# **********************

class Word(Block):
    pass

class Conditional(Word):
    pass

class On(Conditional):
    pass

class Facing(Conditional):
    pass

CONDITION_CLASSES = (On, Facing)


class Conjunction(Word):
    pass

class And(Conjunction):
    pass


CONJUNCTION_CLASSES = (And,)


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

class All(Noun):
    pass


NOUN_CLASSES = (Baba, Flag, Text, Me, Robot, Empty, All)


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
   
    def _rules_gen(self, max_depth: int = 10) -> Iterator[Self]:
        """generate any/all rules found, from left to right"""
        verb_idxs = [i for i, word in enumerate(self.words) if isinstance(word, Verb)]
        start_idxs = [0] + [i+1 for i in verb_idxs]
        verb_idxs.append(self.num_words)
        end_idxs = verb_idxs[1:]

        for start_idx, verb_idx, end_idx in zip(start_idxs, verb_idxs, end_idxs):
            verb = self.words[verb_idx]
            # print(f'{left_idx=}, {verb_idx=}, {end_idx=}, {self.num_words=}')
            for left_idx, right_idx in product(range(start_idx, verb_idx), range(end_idx, verb_idx, -1)):
                left_words = self.words[left_idx:verb_idx]
                right_words = self.words[verb_idx+1:right_idx]
                # print(f'{left_idx=}, {left_words=}')
                # print(f'{right_idx=}, {right_words=}')
                if Sentence(words=left_words).is_subject(max_depth=max_depth) and Sentence(words=right_words).is_object(max_depth=max_depth):
                    yield Sentence(words=left_words + [verb] + right_words)
                    # only one rule can be formed between verbs
                    break
    
    def to_rules(self, max_depth: int = 10) -> list[Self]:
        return list(self._rules_gen(max_depth=max_depth))
    
    def to_rule(self, max_depth: int = 10, idx: int = 0) -> Self | None:
        for i, rule in enumerate(self._rules_gen(max_depth=max_depth)):
            if i == idx:
                return rule
        else:
            return None
        
    def to_subject(self, max_depth: int = 10, idx: int = 0) -> Self | None:
        rule = self.to_rule(max_depth=max_depth, idx=idx)
        if rule is not None:
            for i, word in enumerate(rule.words):
                if isinstance(word, Verb):
                    s = Sentence(words=self.words[:i])
                    assert s.is_subject(max_depth=max_depth)
                    return s
        else:
            return None
    
    def to_object(self, max_depth: int = 10, idx: int = 0) -> Self | None:
        rule = self.to_rule(max_depth=max_depth, idx=idx)
        if rule is not None:
            for i, word in enumerate(rule.words):
                if isinstance(word, Verb):
                    s = Sentence(words=self.words[i+1:])
                    assert s.is_object(max_depth=max_depth)
                    return s
        else:
            return None
        
    def to_verb(self, max_depth: int = 10, idx: int = 0) -> Self | None:
        rule = self.to_rule(max_depth=max_depth, idx=idx)
        if rule is not None:
            for i, word in enumerate(rule.words):
                if isinstance(word, Verb):
                    s = Sentence(words=[word])
                    assert s.is_verb(max_depth=max_depth)
                    return s
        else:
            return None
        
    def to_first_rule(self, max_depth: int = 10) -> Self | None:
        return self.to_rule(max_depth=max_depth, idx=0)
        
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
    
    def is_verb(self):
        # subject means: noun [conj noun [conj noun...]]
        return len(self.words) == 1 and isinstance(self.words[0], Verb)
    
    def is_subject(self, max_depth: int = 10):
        # subject means: noun [conj noun [conj noun...]]
        return self.is_complex(restrict_to_noun_only=True, restrict_to_conj_only=False, max_depth=max_depth)
    
    def is_object(self, max_depth: int = 10):
        # object means: noun|adj [conj noun|adj [conj noun|adj...]]
        return self.is_complex(restrict_to_noun_only=False, restrict_to_conj_only=True, max_depth=max_depth)
    
    def is_complex(self, restrict_to_noun_only: bool = False, restrict_to_conj_only: bool = True, max_depth: int = 10):
        # complex means: simple [conj simple [conj simple...]]
        # or: simple [conditional simple]
        lst = []
        n_found = 0
        for i, word in enumerate(self.words):
            if isinstance(word, Conditional):
                if restrict_to_conj_only or n_found > 0:
                    return False
                elif Sentence(words=lst).is_simple(restrict_to_noun_only=restrict_to_noun_only):
                    lst = self.words[i+1:]
                    break
                else:
                    return False
            elif isinstance(word, Conjunction):
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


class Square(BaseModel):
    loc: Location
    blocks: list[Block] = Field(default_factory=list[Block])
    
    @classmethod
    def from_block_loc(cls, loc: Location, block: Block | None = None) -> Self:
        return Square(loc=loc, blocks=[block] if block is not None else [])

    @classmethod
    def from_block_xy(cls, x: int, y: int, block: Block | None = None) -> Self:
        return cls.from_block_loc(block=block, loc=Location(x=x, y=y))

# GridLine = Annotated[list[Square], conlist(Square, min_length=1)]
# SquareGrid = Annotated[list[GridLine], conlist(GridLine, min_length=1)]
GridLine = conlist(Square, min_length=1)
SquareGrid = conlist(GridLine, min_length=1)
# SquareGrid = npt.NDArray[Square]

class Board(BaseModel):
    """board has a m x n grid where (x, y) coordinates are used to locate squares
    col x in [0, 1, ..., m-1], row y in [0, 1, ..., n-1]"""
    grid: SquareGrid
        
    @classmethod
    def from_shape(cls, height: int, width: int) -> Self:
        return Board(grid=[[Square.from_block_xy(x=x, y=y) for x in range(width)] for y in range(height)])
    
    @property
    def height(self) -> int:
        return len(self.grid)
    
    @property
    def width(self) -> int:
        assert self.height > 0
        return len(self.grid[0])
    
    def get(self, x: int, y: int) -> Square | None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None
        return self.row(y=y)[x]
            
    def add_block(self, x: int, y: int, block: Block) -> bool:
        square = self.get(x=x, y=y)
        if square is not None and all(b is not block for b in square.blocks):
            square.blocks.append(block)
            return True
        return False
    
    def pop_block(self, x: int, y: int, block: Block) -> bool:
        square = self.get(x=x, y=y)
        if square is not None:
            for i, b in enumerate(square.blocks):
                if b is block:
                    square.blocks.pop(i)
                    return True
        return False
    
    def pop_blocks(self, x: int, y: int, block_type: type[Block] = Block) -> list[Block]:
        # remove all blocks of a (sub)type
        square = self.get(x=x, y=y)
        keep = []
        discard = []
        if square is not None:
            for block in square.blocks:
                if isinstance(block, block_type):
                    discard.append(block)
                else:
                    keep.append(block)
        square.blocks = keep
        return discard
    
    def row(self, y: int) -> GridLine:
        assert 0 <= y < self.height
        return self.grid[y]
    
    def rows(self) -> Iterator[GridLine]:
        for row in self.grid:
            yield row
    
    def col(self, x: int) -> GridLine:
        assert 0 <= x < self.width
        return [row[x] for row in self.rows()]

    def cols(self) -> Iterator[GridLine]:
        for x in range(self.width):
            yield self.row(x=x)
    
    def words(self, seq: GridLine) -> Iterator[list[Block]]:
        # given a row or column (contiguous sequence of squares)
        # iterate over all possible maximal contiguous word sequences
        word_blocks_seq = []
        for square in seq:
            word_blocks = [block for block in square.blocks if isinstance(block, Word)]
            if len(word_blocks) > 0:
                word_blocks_seq.append(word_blocks_seq)
            else:
                if len(word_blocks_seq) > 0:
                    for combo in product(*word_blocks_seq):
                        yield combo
                word_blocks_seq = []
        if len(word_blocks_seq) > 0:
            for combo in product(*word_blocks_seq):
                yield combo
            
    def rules(self) -> set[Sentence]:
        rules = set[Sentence]()
        for seq in chain(self.rows(), self.cols()):
            for words in self.words(seq):
                for rule in Sentence(words=words).to_rules():
                    rules.add(rule)
        return rules
    
    def rules_with_verb_adj(self, verb_type: type[Verb] | None, adj_type: type[Adjective] | None) -> list[Sentence]:
        rules = []
        for rule in self.rules():
            # subj = rule.to_subject()
            verb = rule.to_verb()
            if verb_type is not None:
                if not isinstance(verb.words[0], verb_type):
                    continue
            
            if adj_type is not None:
                for word in rule.to_object():
                    if isinstance(word, adj_type):
                        rules.append(rule)
        return rules
    
    def subjects_is_you(self) -> list[Word]:
        subjects = []
        for rule in self.rules_with_verb_adj(verb_type=Is, adj_type=You):
            subj = rule.to_subject()
            if subj is not None:
                # lst = []
                # for word in subj.words:
                #     if isinstance(word, (Conjunction, Conditional)):
                #         subjects.append(lst)
                #         lst = []
                #     else:
                #         lst.append(word)
                if subj.is_simple():
                    subjects.append(subj)
        return subjects
    
    def nouns_is_you(self) -> list[Word]:
        nouns = []
        return nouns
    
    def images_is_you(self) -> list[Image]:
        images = []
        return images
            
    def __repr__(self) -> str:
        left_sep = '['
        right_sep = ']'
        horiz_sep = '\n'
        intra_col_sep = '|'
        top_sep = '_'
        bottom_sep = '-'
        row_strs = []
        max_width = 0
        for row in reversed(list(self.rows())):
            col_strs = []
            for col in row:
                col_str = intra_col_sep.join(b.__repr__() for b in col.blocks)
                col_strs.append(col_str)
            row_str = f'{right_sep}{left_sep}'.join(col_strs)
            if len(row_str) > max_width:
                max_width = len(row_str)
            row_strs.append(row_str)
        full_str = f'{right_sep}{horiz_sep}{left_sep}'.join(row_strs)
        return horiz_sep + top_sep*(max_width+2) + horiz_sep + left_sep + full_str + right_sep + horiz_sep + bottom_sep*(max_width+2)

class Level(BaseModel):
    # history: Annotated[list[Board], conlist(min_length=1)]
    history: conlist(Board, min_length=1)
    
    @property
    def board(self) -> Board:
        return self.history[-1]
    
    @classmethod
    def from_board(cls, board: Board) -> Self:
        return Level(history=[board])
    
    def rules(self) -> list[Sentence]:
        return self.board.rules()
        
    def forward(self, d: Direction) -> None:
        raise NotImplementedError()
    
    def backward(self) -> None:
        if len(self.history) > 1:
            self.history.pop()
            
    def step(self, pressed: Direction | None) -> None:
        if isinstance(pressed, Direction):
            self.forward(pressed)
        else:
            self.backward()
            
    
if __name__ == "__main__":
    s = Sentence(words=[You(), Is(), Baba(), And(), Robot(), Is(), Push(), And(), You(), Is(), Baba()])
    print(f'{s=}')
    r = s.to_first_rule()
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
    
    print(f'{(Baba() == Baba())=}')
    print(f'{(Baba() is Baba())=}')
    
    board = Board.from_shape(height=5, width=6)
    board.add_block(y=1, x=1, block=Baba())
    board.add_block(y=1, x=2, block=Is())
    board.add_block(y=1, x=3, block=You())
    print(f'{board=}')
    level = Level.from_board(board=board)
    print(f'{level=}')
