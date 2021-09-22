import json
import os
from collections import defaultdict
import networkx as nx
import numpy as np
import copy
import pprint
pp = pprint.PrettyPrinter(indent=4)
import sys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
from pathlib import Path


def load_nav_graphs(scans, connectivity):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    # connect_folder = rootDir
    for scan in scans:
        data = connectivity['%s_connectivity'%scan]
        G = nx.Graph()
        positions = {}

        for i,item in enumerate(data):
            if item['included']:
                for j,conn in enumerate(item['unobstructed']):
                    if conn and data[j]['included']:
                        positions[item['image_id']] = np.array([item['pose'][3],
                                item['pose'][7], item['pose'][11]]);
                        assert data[j]['unobstructed'][i], 'Graph should be undirected'
                        G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
        nx.set_node_attributes(G, values=positions, name='position')
        graphs[scan] = G
    return graphs

class Evaluation(object):
    def __init__(self, anno):
        # self.gt = anno["gt"]
        self.error_margin = 3.0
        self.connectivity = anno["connectivity"]
        self.gt = {}
        self.instr_ids = []
        self.scans = []

        for item in anno["gt"]:
            scan = item['bboxes'][0]['scan']
            if scan not in self.scans:
                self.scans.append(scan)
            self.gt[str(item['path_id'])] = copy.deepcopy(item)
            new_instrs = []
            instructions = copy.deepcopy(item['instructions'])
            for i in range(len(instructions)):
                new_instrs.append(instructions[i][4])
            self.gt[str(item['path_id'])]['instructions'] = new_instrs
        
        self.graphs = load_nav_graphs(self.scans, self.connectivity)
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
        import ipdb; ipdb.set_trace()

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id
    
    def _score_item(self, instr_id, path, heading=None, elevation=None, pre_bbox=None, pre_num_heading=None, pre_num_elevation=None):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        heading = np.array(heading)
        elevation = np.array(elevation)
        gt = self.gt[instr_id.split('_')[-2]]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]    # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)

        pre_point = Point(heading, elevation)
        # pre_point = np.array([heading, elevation]).reshape(1, 2)
        success = False
        for bbox in gt['bboxes']:
            if bbox['image_id'] == final_position:
                goal = final_position
                gt_heading = bbox['heading']
                gt_elevation = bbox['elevation']
                # gt_point = Point(gt_heading, gt_elevation)
                gt_poly = Polygon([(bbox['target']['left_top']['heading'], bbox['target']['left_top']['elevation']),
                                    (bbox['target']['right_top']['heading'], bbox['target']['right_top']['elevation']),
                                    (bbox['target']['right_bottom']['heading'], bbox['target']['right_bottom']['elevation']),
                                    (bbox['target']['left_bottom']['heading'], bbox['target']['left_bottom']['elevation'])])

                self.scores['heading_errors'].append(math.fabs((gt_heading - heading)))
                self.scores['elevation_errors'].append(math.fabs((gt_elevation - elevation)))
                if gt_poly.contains(pre_point):
                    # point_inds = (gt_poly.contains_points(pre_point) > 0).nonzero()[0]
                    # if point_inds.shape[0] > 0:
                    self.scores['det_success_num'].append(1.)
                    # self.scores['point_det_errors'].append(pre_point.distance(gt_point))
                    self.scores['point_det_errors'].append(math.hypot(gt_heading - heading, gt_elevation - elevation))
                    success = True
                break
        if not success:
            self.scores['det_success_num'].append(0.)

        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])

        self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_lengths'].append(
            self.distances[gt['scan']][start][goal]
        )
        self.scores['goal_progress'].append(
            self.distances[gt['scan']][start][goal] - self.distances[gt['scan']][final_position][goal]
        )

    def score(self, results):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        for item in results:
            self._score_item(item['instr_id'], item['trajectory']['path'],
                                item['trajectory']['obj_heading'],
                                item['trajectory']['obj_elevation'])

        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'lengths': np.average(self.scores['trajectory_lengths'])
        }
        score_summary['heading_errors'] = np.average(self.scores['heading_errors'])
        score_summary['elevation_errors'] = np.average(self.scores['elevation_errors'])
        score_summary['point_det_errors'] = np.average(self.scores['point_det_errors'])
        score_summary['goal_progress'] = np.average(self.scores['goal_progress'])

        det_num_successes = len([i for i in self.scores['det_success_num'] if i > 0.])
        score_summary['det_success_rate'] = float(det_num_successes) / float(len(self.scores['det_success_num']))
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        score_summary['nav_success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))

        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))

        spl = [float(error < self.error_margin) * l / max(l, p, 0.01)
            for error, p, l in
            zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['spl'] = np.average(spl)
        success_rate = np.array(self.scores['det_success_num']) * np.array(spl)
        score_summary['success_rate'] = np.average(success_rate)
        return score_summary

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.........")
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata']) 
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    output = {}

    print(f"load {test_annotation_file} {user_submission_file}")

    print(f"test_annotation_file size: {Path(test_annotation_file).stat().st_size}")
    print(f"user_submission_file size: {Path(user_submission_file).stat().st_size}")

    with open(test_annotation_file,'r') as f:
        f_str = f.read()
        anno = json.loads(f_str)

    with open(user_submission_file,'r') as f:
        f_str = f.read()
        submit_data = json.loads(f_str)

    print("load finish")

    print("Evaluating for %s Phase" % phase_codename)
    ev = Evaluation(anno)
    print("init finished")
    score_summary = ev.score(submit_data)
    print("score finished")

    output["result"] = [
        {
            "%s_split" % phase_codename: {
                "length": round(score_summary['lengths'],2),
                "SR": round(score_summary['nav_success_rate'],2),
                "OSR": round(score_summary['oracle_rate'],2),
                "SPL": round(score_summary['spl'],2),
                "SFPL": round(score_summary['success_rate'],2)
            }
        },
    ]
    
    # To display the results in the result file
    output["submission_result"] = output["result"][0]
    # output = {'result': [{'test_split': {'length': 17.79, 'SR': 0.08, 'OSR': 0.11, 'SPL': 0.07, 'SFPL': 0.01}}], 'submission_result': {'test_split': {'length': 17.79, 'SR': 0.08, 'OSR': 0.11, 'SPL': 0.07, 'SFPL': 0.01}}}
    print(output)
    print("Completed evaluation for %s" % phase_codename)
    return output
