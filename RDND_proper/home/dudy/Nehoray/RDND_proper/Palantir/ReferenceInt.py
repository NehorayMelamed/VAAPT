class ReferenceInt:
    # the point of this class is that it can be passed in a function, and a reference is passed.
    # Needed for results saver saving of PNGs. Not a very complete class.
    # for example, say I were to do this:
    #
    # def add_one(num):
    #   num+=1
    # regular = 0
    # ref = ReferenceInt(0)
    # add_one(regular)
    # add_one(ref)
    #
    # regular will still equal 0, but ref will equal 1

    def __init__(self, val=0):
        self.val = val

    def __iadd__(self, other):
        if type(other) == int or type(other) == float:
            self.val += int(other)
            return self
        else:
            self.val += other.val
            return self

    def __add__(self, other):
        self.__iadd__(other)
        return ReferenceInt(self.val)

    def __str__(self):
        return str(self.val)

    def __int__(self):
        return self.val


if __name__ == '__main__':
    # debugging
    a = ReferenceInt(0)
    a+=5
    print(a)
