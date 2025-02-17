import os
import json
import random
import argparse


def show_score(datas, score_file_path):
    score_datas = [json.loads(line) for line in open(score_file_path)]
    id_2_score_data = {score_data['id']: score_data for score_data in score_datas}
    if len(datas) != len(score_datas):
        print("==================WARNING==================")
        print(f"len(datas) != len(score_datas): {len(datas)} != {len(score_datas)}")
        print("===========================================")
    
    ok_num = 0
    error_num = 0
    ok_score = {"Analysis Score": 0, "Technology Score": 0}
    for data in datas:

        score_data = id_2_score_data[data['id']]

        if score_data['score'] == None:
            error_num += 1
            print(f"None score, score_data['id']: {score_data['id']}")
            continue
        ok_num += 1
        ok_score["Analysis Score"] += score_data['score']['Analysis Score']
        ok_score["Technology Score"] += score_data['score']['Technology Score']

    print(f"Ok num: {ok_num}, Error num: {error_num}")
    print(f"Average score: Analysis Score: {ok_score['Analysis Score']/ok_num:.1f}, Technology Score: {ok_score['Technology Score']/ok_num:.1f}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='en')
    args = parser.parse_args() 

    for scenario in [
        '0_test',
        # '1_environment',
        # '2_mining', 
        # '3_transport',
        # '4_aerospace',
        # '5_telecom',
        # '6_architecture',
        # '7_water',
        # '8_farming',
    ]:
        datas = json.load(open(f"benchmark/{scenario}/datas.json"))
        print("\n\n\n")

        print(f"{scenario}")

        score_file_path = f"./eval_results/{scenario}/generated_by_5_ours_5_2_1_.jsonl_score.jsonl"

        show_score(datas, score_file_path)

        print("Done.")