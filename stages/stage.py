
class StageInterface:

    def __init__(self, concat_output=False, visualize=False):
        self.concat_output = concat_output
        self.visualize = visualize

    def execute(self, **kwargs):
        results = self._execute(**kwargs)
        if self.visualize:
            self._visualize()
        if self.concat_output:
            return self._concat_output(results)
        return results

    def _execute(self, inp, meta, **kwargs):
        pass

    def _concat_output(self, outp):
        pass

    def _visualize(self):
        pass