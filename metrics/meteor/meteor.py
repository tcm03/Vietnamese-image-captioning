import os
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py. Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


class Meteor:

    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR,
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         encoding='utf-8')  # Set encoding to UTF-8
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert list(gts.keys()) == list(res.keys())
        imgIds = list(gts.keys())
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        try:
            for i in imgIds:
                assert len(res[i]) == 1
                stat = self._stat(res[i][0], gts[i])
                eval_line += f' ||| {stat}'

            self.meteor_p.stdin.write(f'{eval_line}\n')
            self.meteor_p.stdin.flush()
            for _ in range(len(imgIds)):
                scores.append(float(self.meteor_p.stdout.readline().strip()))
            score = float(self.meteor_p.stdout.readline().strip())
        finally:
            self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = f'SCORE ||| {" ||| ".join(reference_list)} ||| {hypothesis_str}'
        self.meteor_p.stdin.write(f'{score_line}\n')
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        try:
            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            score_line = f'SCORE ||| {" ||| ".join(reference_list)} ||| {hypothesis_str}'
            self.meteor_p.stdin.write(f'{score_line}\n')
            self.meteor_p.stdin.flush()
            stats = self.meteor_p.stdout.readline().strip()
            eval_line = f'EVAL ||| {stats}'
            # EVAL ||| stats
            self.meteor_p.stdin.write(f'{eval_line}\n')
            self.meteor_p.stdin.flush()
            score = float(self.meteor_p.stdout.readline().strip())
            # Bug fix: there are two values returned by the jar file, one average, and one all
            score = float(self.meteor_p.stdout.readline().strip())
        finally:
            self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        try:
            if self.meteor_p.stdin:
                self.meteor_p.stdin.close()
            if self.meteor_p:
                self.meteor_p.kill()
                self.meteor_p.wait()
        finally:
            self.lock.release()
