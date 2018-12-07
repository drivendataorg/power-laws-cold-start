from collections import defaultdict


class find_union(object):

    def __init__(self, reduce_fn=None):
        self.parent = {}
        self.set_size = defaultdict(int)
        self.extra_values = {}
        self.reduce_fn = reduce_fn

    def exists(self, x):
        return x in self.parent

    def create_element(self, x, extra_values=None):
        assert not self.exists(x)
        self.parent[x] = x
        self.set_size[x] = 1
        self.extra_values[x] = extra_values

    def create_element_if_not_exists(self, x, extra_values=None):
        if not self.exists(x):
            self.create_element(x, extra_values)

    def update_element(self, x, extra_values=None):
        assert self.exists(x)
        x1 = self.find(x)
        if self.reduce_fn is not None:
            self.extra_values[x1] = self.reduce_fn(self.extra_values[x1], extra_values)
        else:
            self.extra_values[x1] = extra_values

    def find(self, x):
        if not self.exists(x):
            self.create_element(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def get_extra_value(self, x):
        x1 = self.find(x)
        return self.extra_values[x1]

    def union(self, x, y):
        x1 = self.find(x)
        y1 = self.find(y)
        if x1 != y1:
            sx = self.set_size[x1]
            sy = self.set_size[y1]
            if sx > sy:
                x1, y1, sx, sy = y1, x1, sy, sx
            self.parent[x1] = y1
            self.set_size[y1] += sy
            if self.reduce_fn is not None:
                self.extra_values[y1] = self.reduce_fn(self.extra_values[y1], self.extra_values[x1])


def main():
    fu = find_union(reduce_fn=lambda x, y: (x[0] + y[0], x[1] + y[1]))
    fu.create_element(1, (1, 1))
    fu.create_element(2, (5, 10))
    assert fu.get_extra_value(1) == (1, 1)
    fu.union(1, 2)
    assert fu.get_extra_value(1) == (6, 11)
    assert fu.get_extra_value(2) == (6, 11)
    fu.update_element(1, (1, 1))
    assert fu.get_extra_value(1) == (7, 12)
    print("OK")


if __name__ == '__main__':
    main()