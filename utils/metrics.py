from _3rdparty import py_sod_metrics


class FullMetrics:
    def __init__(self) -> None:
        self.FM = py_sod_metrics.Fmeasure()
        self.WFM = py_sod_metrics.WeightedFmeasure()
        self.SM = py_sod_metrics.Smeasure()
        self.EM = py_sod_metrics.Emeasure()
        self.MAE = py_sod_metrics.MAE()
        self.MSIOU = py_sod_metrics.MSIoU()
    
    def eval_step(self, pred, mask):
        self.FM.step(pred=pred, gt=mask)
        self.WFM.step(pred=pred, gt=mask)
        self.SM.step(pred=pred, gt=mask)
        self.EM.step(pred=pred, gt=mask)
        self.MAE.step(pred=pred, gt=mask)
        self.MSIOU.step(pred=pred, gt=mask)
        
    def result_all(self):
        fm = self.FM.get_results()["fm"]
        wfm = self.WFM.get_results()["wfm"]
        sm = self.SM.get_results()["sm"]
        em = self.EM.get_results()["em"]
        mae = self.MAE.get_results()["mae"]
        msiou = self.MSIOU.get_results()["msiou"]

        return {
            "MAE": mae,
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MSIOU": msiou,
            # E-measure for sod
            "adpEm": em["adp"],
            "meanEm": em["curve"].mean(),
            "maxEm": em["curve"].max(),
            # F-measure for sod
            "adpFm": fm["adp"],
            "meanFm": fm["curve"].mean(),
            "maxFm": fm["curve"].max()
        }