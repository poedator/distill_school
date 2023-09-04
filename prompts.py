"""functions to get CoT fewshot prompts from popular papers"""

import os, sys

cot_hub_dir = '/home/optimus/chain_of_thought_hub/gsm8k/lib_prompt'

def get_cot_hub_prompt(prompt_filename='prompt_original.txt', raw=False):

    with open(os.path.join(cot_hub_dir, prompt_filename)) as pf:
        prompt_cot_0 = pf.read()
        
    if raw:
        return prompt_cot_0
    
    # replace "\n" with ";" in the steps+answer block, so \n becomes a stop token
    prompt_cot = prompt_cot_0 \
        .strip() \
        .replace('\n\nQuestion', '|Question') \
        .replace('\nLet', '|Let') \
        .replace('\n', ' ') \
        .replace('|', '\n') \
        .replace('step by step', 'step by step:') \
        .replace('tues- day', 'tuesday')
    # print(*prompt_cot.split('\n'), sep = '\n>>>')

    return prompt_cot



def get_kojima_prompt(cot_flag, dataset='gsm8k', direct_answer_trigger_for_fewshot="The answer is"):
    x, z, y = [], [], []
    
    # example sentences ...    
    if dataset in ("multiarith", "gsm8k"):
        
        x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")        

        x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")        

        x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        y.append("8")        

        x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        y.append("9")        

        x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")        

        x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")        

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        y.append("8")
    
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # randomize order of the examples ...
    index_list = list(range(len(x)))
    # random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    
    return demo_text

kojima_prompt_cot = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."""

kojima_prompt_nocot = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: The answer is 8."""

cot_hub_prompt_original_1 = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Let's think step by step: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Let's think step by step: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Let's think step by step: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Let's think step by step: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Let's think step by step: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Let's think step by step: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Let's think step by step: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let's think step by step: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."""