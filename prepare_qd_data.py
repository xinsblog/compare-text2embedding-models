import pandas as pd
import tqdm
import requests


def ask_chatglm(prompt, url):
    resp = requests.post(url, json={
        'prompt': prompt,
        'history': []
    })
    return resp.json()['response']


if __name__ == '__main__':
    import sys
    chatglm_api_url = sys.argv[1]

    qq_df = pd.read_csv("data/qq.csv")
    output_data = []
    for index, row in tqdm.tqdm(qq_df.iterrows()):
        q1, q2 = row['q1'], row['q2']
        d1 = ask_chatglm(q1, chatglm_api_url)
        d2 = ask_chatglm(q2, chatglm_api_url)
        output_data.append((q1, d1, '1'))
        output_data.append((q2, d2, '1'))
        output_data.append((q1, d2, '0'))
        output_data.append((q2, d1, '0'))

    qd_df = pd.DataFrame(
        output_data, columns=['q', 'd', 'label']
    )
    qd_df.to_csv("data/qd.csv", index=False, encoding='utf-8')
