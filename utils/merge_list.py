from __future__ import annotations


class MergeList(list):
    _set = []

    def __init__(self, o: Optional[list] = None):
        if o:
            super().__init__(o)
            self._set = [hash(u) for u in o]

    def append(self, o: MergeList | list | str | int):
        if not hasattr(o, "__iter__"):
            super().append(o)
            self._set.append(hash(o))
            return

        if isinstance(o, list):
            o = MergeList(o)

        for h, u in zip(o._set, o):
            if not h in self._set:
                self._set.append(h)
                super().append(u)

    def __iadd__(self, o: MergeList | list):
        self.append(o)
        return self
