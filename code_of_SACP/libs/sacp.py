import torch

from .aps import APS
from .raps import RAPS
from .saps import SAPS
from torchcp.utils.common import calculate_conformal_value
from torchcp.classification.utils.metrics import Metrics
from torchcp.classification.utils import ConfCalibrator
from .config import parameters
from .utils import get_device


class SACP():
    def __init__(self, 
        model, 
        model_name,
        model_dir, 
        data_name, 
        base_score, 
        data_dict, 
        alpha=0.05, 
        lmd=0.5, 
        weight=0.02,
        k=1,
        ss=[[0, 5]]
    ):
        super(SACP, self).__init__()
        self.model = model
        self.base_score = base_score
        self.model_name = model_name
        self.model_dir = model_dir
        self.data_name = data_name
        self.calibloader = data_dict['calibloader']
        self.testloader = data_dict['testloader']
        self.h = data_dict['h']
        self.w = data_dict['w']
        self.lmd = lmd
        self.k = k
        self.ss = ss
        self.alpha = alpha
        self.weight = weight
        self.device = get_device()
        self.parameters = parameters
        self.cal_indices = data_dict["cal_indices"]
        self.test_indices = data_dict["test_indices"]

    
    def _generate_prediction_set(self, scores, q_hat):
        return [torch.argwhere(scores[i] < q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]

    def predict_with_logits(self, scores, q_hat=None):
        S = self._generate_prediction_set(scores, q_hat)
        return S

    def get_scores(self, score_function, logits, label=None):
        return score_function(logits, label)

    def fusion_scores(self, score_map, index):
        h = score_map.shape[0]
        w = score_map.shape[1]
        row = index // w
        col = index % w
        d = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
        cnt = 0.0
        ori_score = score_map[row, col]
        neighbor_score = torch.zeros_like(ori_score)
        for j in d:
            t_row = row + j[0]
            t_col = col + j[1]
            if t_row >=0 and t_row <= h - 1 and t_col >= 0 and t_col <= w - 1:
                if score_map[t_row, t_col, 0] != 0:
                    neighbor_score += score_map[t_row, t_col]
                    cnt += 1
        fusion_score = self.lmd * ori_score + self.lmd * (neighbor_score / cnt)
        return fusion_score

    def calculate_sacp(self):
        h = self.h
        w = self.w
        pre_dict = self.predict()
        cal_logits = pre_dict["cal_logits"]
        cal_labels = pre_dict["cal_labels"]
        test_logits = pre_dict["test_logits"]
        test_labels = pre_dict["test_labels"]

        if self.base_score == 'APS':
            all_cal_base_scores = self.get_scores(APS(), cal_logits, label=cal_labels)
            cal_base_scores = self.get_scores(APS(), cal_logits)
            test_base_scores = self.get_scores(APS(), test_logits)
        elif self.base_score == 'RAPS':
            p = self.parameters[self.model_name][self.data_name]['penalty']
            all_cal_base_scores = self.get_scores(RAPS(p), cal_logits, label=cal_labels)
            cal_base_scores = self.get_scores(RAPS(p), cal_logits)
            test_base_scores = self.get_scores(RAPS(p), test_logits)
        elif self.base_score == 'SAPS':
            all_cal_base_scores = self.get_scores(SAPS(self.weight), cal_logits, label=cal_labels)
            cal_base_scores = self.get_scores(SAPS(self.weight), cal_logits)
            test_base_scores = self.get_scores(SAPS(self.weight), test_logits)
        else:
            raise ValueError("This base score is not supported.")

        base_q_hat = calculate_conformal_value(all_cal_base_scores, self.alpha)
        base_prediction_sets = self.predict_with_logits(test_base_scores, base_q_hat)

        base_map = torch.zeros((h * w, cal_base_scores.shape[1]))
        for e, i in enumerate(self.cal_indices):
            base_map[i] = cal_base_scores[e]
        for e, i in enumerate(self.test_indices):
            base_map[i] = test_base_scores[e]

        base_map = torch.reshape(base_map, (h, w, cal_base_scores.shape[1]))
        score_map_list = []
        score_map_list.append(base_map)

        for map_i in range(self.k):
            score_map = score_map_list[map_i]
            new_map = torch.zeros_like(score_map)
            for _, i in enumerate(self.cal_indices):
                row = i // w
                col = i % w
                fusion_score = self.fusion_scores(score_map, i)
                new_map[row, col] = fusion_score
            for _, i in enumerate(self.test_indices):
                row = i // w
                col = i % w
                fusion_score = self.fusion_scores(score_map, i)
                new_map[row, col] = fusion_score
            score_map_list.append(new_map)

        fusion_score_map = torch.reshape(score_map_list[-1], (h * w, cal_base_scores.shape[1]))

        fusion_cal_scores = torch.zeros(cal_base_scores.shape[0])
        for e, i in enumerate(self.cal_indices):
            fusion_cal_scores[e] = fusion_score_map[i, int(cal_labels[e])]
        
        fusion_cal_scores = torch.nan_to_num(fusion_cal_scores, nan=0.0)
        fusion_q_hat = calculate_conformal_value(fusion_cal_scores, self.alpha)

        fusion_test_scores = torch.zeros_like(test_base_scores)
        for e, i in enumerate(self.test_indices):
            fusion_test_scores[e] = fusion_score_map[i]
        

        fusion_prediction_sets = self.predict_with_logits(fusion_test_scores, fusion_q_hat)

        metric = Metrics()
        res_dict = {
            'cov' : metric('coverage_rate')(base_prediction_sets, test_labels),
            'size' : metric('average_size')(base_prediction_sets, test_labels),
            'SSCV' : 100 * metric('SSCV')(base_prediction_sets, test_labels, self.alpha, stratified_size=self.ss),
            'fusion_cov' : metric('coverage_rate')(fusion_prediction_sets, test_labels),
            'fusion_size' : metric('average_size')(fusion_prediction_sets, test_labels),
            'fusion_SSCV' : 100 * metric('SSCV')(fusion_prediction_sets, test_labels, self.alpha, stratified_size=self.ss),
        }
        return res_dict

    def predict(self):
        self.model.load_state_dict(torch.load(self.model_dir))
        self.model.eval()
        self.model = self.model.to(self.device)

        test_labels= []
        test_logits = []
        tem = self.parameters[self.model_name][self.data_name]['tem']
        logits_transformation = ConfCalibrator.registry_ConfCalibrator("TS")(tem)
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                test_labels.append(labels)
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = logits_transformation(self.model(images.float())).detach()
                test_logits.append(outputs)

        cal_labels = []
        cal_logits = []

        with torch.no_grad():
            for data in self.calibloader:
                images, labels = data
                cal_labels.append(labels)
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = logits_transformation(self.model(images.float())).detach()
                cal_logits.append(outputs)

        pre_dict = {
            "cal_logits" : torch.cat(cal_logits),
            "cal_labels" : torch.cat(cal_labels).to(self.device),
            "test_logits" : torch.cat(test_logits),
            "test_labels" : torch.cat(test_labels).to(self.device),
        }
        return pre_dict



    

        
