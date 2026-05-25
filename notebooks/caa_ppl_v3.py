# %%
import pandas as pd

# %%
gemma_1b_instruct_ppl = pd.read_json("../data/v3_ppl_json/Gemma3_1B_Instruct/ppl_results.json").transpose()
qwen_0_5b_instruct_ppl = pd.read_json("../data/v3_ppl_json/Qwen2_5_0_5B_Instruct/ppl_results.json").transpose()
qwen_1_5b_instruct_ppl = pd.read_json("../data/v3_ppl_json/Qwen2_5_1_5B_Instruct/ppl_results.json").transpose()
qwen_3b_instruct_ppl = pd.read_json("../data/v3_ppl_json/Qwen2_5_3B_Instruct/ppl_results.json").transpose()
qwen_7b_instruct_ppl = pd.read_json("../data/v3_ppl_json/Qwen2_5_7B_Instruct/ppl_results.json").transpose()
tulu_8b_instruct_ppl = pd.read_json("../data/v3_ppl_json/Llama3_1_Tulu_3_1_8B/ppl_results.json").transpose()


# %%
gemma_1b_instruct_ppl.to_csv("../data/v3_csv/ppl/gemma_1b_instruct_ppl.csv")
qwen_0_5b_instruct_ppl.to_csv("../data/v3_csv/ppl/qwen_0_5b_instruct_ppl.csv")
qwen_1_5b_instruct_ppl.to_csv("../data/v3_csv/ppl/qwen_1_5b_instruct_ppl.csv")
qwen_3b_instruct_ppl.to_csv("../data/v3_csv/ppl/qwen_3b_instruct_ppl.csv")
qwen_7b_instruct_ppl.to_csv("../data/v3_csv/ppl/qwen_7b_instruct_ppl.csv")
tulu_8b_instruct_ppl.to_csv("../data/v3_csv/ppl/tulu_8b_instruct_ppl.csv")


# %%
qwen_3b_instruct_ppl


