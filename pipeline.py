
class Pipeline:

    def __init__(self, figures_path=None):
        # List of stages
        self.stages = []
        self.figures_path = figures_path

    def add_stage(self, stage, **kwargs):
        self.stages.append((stage, kwargs))

    def execute(self, inp, meta):
        stage_results = [inp]
        for stage, kwargs in self.stages:
            result = stage.execute(inp=stage_results[-1], meta=meta, **kwargs)
            stage_results.append(result)
        return stage_results
