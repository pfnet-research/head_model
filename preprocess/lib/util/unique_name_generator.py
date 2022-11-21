class UniqueNameGenerator(object):

    def __init__(self):
        self.counts = {}

    def make_unique(self, name):
        name = str(name)
        if name in self.counts:
            cnt = self.counts[name]
            self.counts[name] += 1
            return name + '_' + str(cnt)
        else:
            self.counts[name] = 1
            return name
