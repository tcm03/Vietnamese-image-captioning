#!/usr/bin/env python3

# Python wrapper for METEOR implementation, modified for Python 3
# Original code by Xinlei Chen
# Acknowledges Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import threading

# Assumes meteor-1.5.jar is in the same directory as meteor.py. Change as needed.
METEOR_JAR = 'meteor-1.5.jar'

class Meteor:

    def __init__(self):
        self.meteor_cmd = [
            'java', '-Dfile.encoding=UTF-8', '-jar', '-Xmx2G', METEOR_JAR,
            '-', '-', '-stdio', '-l', 'en', '-norm'
        ]
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert set(gts.keys()) == set(res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert len(res[i]) == 1
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode('utf-8'))
        self.meteor_p.stdin.flush()
        for _ in imgIds:
            score_line = self.meteor_p.stdout.readline()
            score_line = score_line.decode('utf-8').strip()
            scores.append(float(score_line))
        total_score_line = self.meteor_p.stdout.readline()
        total_score_line = total_score_line.decode('utf-8').strip()
        score = float(total_score_line)
        self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(
            ('SCORE', ' ||| '.join(reference_list), hypothesis_str)
        )
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode('utf-8'))
        self.meteor_p.stdin.flush()
        stat_line = self.meteor_p.stdout.readline()
        stat_line = stat_line.decode('utf-8').strip()
        return stat_line

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(
            ('SCORE', ' ||| '.join(reference_list), hypothesis_str)
        )
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode('utf-8'))
        self.meteor_p.stdin.flush()
        stats = self.meteor_p.stdout.readline()
        stats = stats.decode('utf-8').strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode('utf-8'))
        self.meteor_p.stdin.flush()
        # Read two lines as per the original code's bug fix
        score_line = self.meteor_p.stdout.readline()
        score_line = score_line.decode('utf-8').strip()
        score = float(score_line)
        score_line = self.meteor_p.stdout.readline()
        score_line = score_line.decode('utf-8').strip()
        score = float(score_line)
        self.lock.release()
        return score

    def __del__(self):
        self.lock.acquire()
        try:
            self.meteor_p.stdin.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
        except Exception:
            pass
        self.lock.release()
