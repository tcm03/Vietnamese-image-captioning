from metrics.bleu.bleu import Bleu
from metrics.cider.cider import Cider
from metrics.tokenizer.ptbtokenizer import PTBTokenizer


class KTViC:
    def __init__(self, json_data: dict):
        # json_data is array of objects where "image_id" is the key
        self.dataset = json_data

    def getImgIds(self):
        return [img["image_id"] for img in self.dataset]

    def imgToAnns(self, imgId):
        for img in self.dataset:
            if img["image_id"] == imgId:
                return img["caption"]
        return None


class KTVICEvalCap:
    def __init__(self, ktvic, ktvicRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.ktvic = ktvic
        self.ktvicRes = ktvicRes
        self.params = {"image_id": ktvic.getImgIds()}

    def evaluate(self):
        imgIds = self.params["image_id"]
        gts = {}  # ground truth
        res = {}  # result

        for imgId in imgIds:
            gts[imgId] = self.ktvic.imgToAnns[imgId]
            res[imgId] = self.ktvicRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================

        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(),"METEOR"),
            # (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print("computing %s score..." % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for _, eval in self.imgToEval.items()]
