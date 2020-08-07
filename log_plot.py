import os
import argparse
import matplotlib.pyplot as plt



def read_code_search_log(path=None):
    file_path = path
    score_strs = [' acc =', ' f1 =']

    result = dict()
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            for score_str in score_strs:
                if line.find(score_str) != -1:
                    score_name = score_str.split('=')[0].strip()
                    if score_name not in result:
                        socre_list = []
                    else:
                        socre_list = result[score_name]
                    socre_list.append(round(float(line.split(score_str)[1]), 3))
                    result[score_name] = socre_list
    draw_plot(result)

def read_code_generation_log(path=None):
    file_path = path
    score_strs = ['train_loss = ', 'eval_ppl = ', "bleu-4 = "]

    result = dict()
    with open(file_path, encoding='utf-8') as f:
        for line in f.readlines():
            for score_str in score_strs:
                if line.find(score_str) != -1:
                    score_name = score_str.split('=')[0].strip()
                    if score_name not in result:
                        socre_list = []
                    else:
                        socre_list = result[score_name]
                    socre_list.append(round(float(line.split(score_str)[1]), 3))
                    result[score_name] = socre_list
    # {"loss": [....], "ppl": [...], "bleu": [...]}
    return result

def draw_plot(result):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--log_path", default=None, type=str, 
    #                     help="The log filename. Should contain the .text files for this task.") 

    # args = parser.parse_args()
    # if args.log_path is not None: 
    #     file_path = args.file_path
    # else:
    #     file_path = path

    for score_str in result.keys():
        plt.plot(range(len(result[score_str])), result[score_str], label=score_str)
    plt.legend()
    plt.show()

class CodeGenerationResult:
    def __init__(self, file_name, loss_list=[], ppl_list=[], bleu_list=[]):
        self.file_name = file_name
        self.loss_list = loss_list
        self.ppl_list = ppl_list
        self.bleu_list = bleu_list

def code_generation_plot():
    code_generation_log_path = r'F:\AIForProgram\CodeBERT\code2nl\log\\'
    results = {}
    file_names = ["eval=300 train=3000", "eval=600 train=20000", "eval=1000 train=50000"]
    for file_name in file_names:
        score_result = read_code_generation_log(os.path.join(code_generation_log_path, file_name + ".txt"))
        # {"loss": [....], "ppl": [...], "bleu": [...]}
        for score_str in score_result.keys(): 
            if score_str in results:
                score_dict = results[score_str]
            else:
                score_dict = {}
            score_dict[file_name] = score_result[score_str]
            # {"loss": {'file1': [....], 'file2':[...] }, "ppl": {...}, "bleu": {...}}
            results[score_str] = score_dict

    for score_str in results.keys():
        plt.figure()
        plt.xlabel('epochs')
        plt.ylabel(score_str)
        for file_name, score_list in results[score_str].items():
            plt.plot(range(len(score_list)), score_list, label=file_name)
        plt.legend()
        plt.savefig(score_str, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # code_search_log_path = r'F:\AIForProgram\CodeBERT\log\\'
    # dataset1 = '400-8-epochs5.txt'
    # dataset2 = 'wikisql-200--BZ20-epochs3.txt'
    # result = read_code_search_log(os.path.join(code_search_log_path, dataset2))
    # draw_plot(result)

    code_generation_plot()
