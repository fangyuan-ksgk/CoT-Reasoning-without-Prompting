from tqdm import tqdm
from typing import List
import random
from generator import simulate_conversation, collect_ideas, generate_revision_ideas

# Single Term Synthetic Conversation Rewritter
class SyntheticRewritter:
    def __init__(self, conversation_seed, revision_goal, n = 3): # n is controlling how many rewrites are applied here
        self.conversation_seed = conversation_seed
        self.revision_goal = revision_goal
        self.revision_ideas = []
        self.conversation_list = []
        self.num_of_ideas = n

    def generate_revision_ideas(self):
        from gen_prompt import revision_idea_prompt
        revision_idea_generation_prompt = revision_idea_prompt.format(conversation=self.conversation_seed, revision_goal=self.revision_goal, num_of_ideas=self.num_of_ideas)
        self.revision_ideas = generate_revision_ideas(revision_idea_generation_prompt)

    def rewrite_conversation(self):
        if not self.revision_ideas:
            print("No revision ideas generated yet.")
            return
        self.conversation_list = []
        from gen_prompt import script_rewriter
        for idea in tqdm(self.revision_ideas, desc="Rewriting Conversations based on revision ideas"):
            rewrite = script_rewriter(idea, self.conversation_seed)
            output_string = rewrite['content']
            try:
                revised_conversation = eval(output_string)
                if len(revised_conversation)>=2:
                    self.conversation_list.append(revised_conversation)
            except:
                continue
            
    def get_conversation_list(self):
        return self.conversation_list
    

class IterativeSyntheticRewritter:
    def __init__(self, conversation_seeds, positive_revision_goal, negative_revision_goal, n=4):
        self.conversation_seeds = conversation_seeds
        self.positive_revision_goal = positive_revision_goal
        self.negative_revision_goal = negative_revision_goal
        self.initialize_conversation()

    def initialize_conversation(self):
        self.conversation_list = []   
        self.positive_revision_ideas = []     
        self.negative_revision_ideas = []
        self.positive_revisions = []
        self.negative_revisions = []    
        if len(self.conversation_list) == 0 and isinstance(self.conversation_seeds[0], list):
            for seed in self.conversation_seeds:
                self.conversation_list.append(seed)
        elif len(self.conversation_list) == 0 and isinstance(self.conversation_seeds, list):
            self.conversation_list.append(self.conversation_seeds)

        self.iteration = 0
        self.positive_pool = self.conversation_list.copy()
        self.negative_pool = self.conversation_list.copy()

    def single_iteration(self):
        tqdm.write("Starting a new iteration...")
        # Sample one seed from positive/negative pool as seeds
        positive_seed = random.choice(self.positive_pool) if self.positive_pool else None
        negative_seed = random.choice(self.negative_pool) if self.negative_pool else None

        tqdm.write(f"Selected positive seed: {positive_seed}")
        tqdm.write(f"Selected negative seed: {negative_seed}")

        positive_rewritter = SyntheticRewritter(positive_seed, self.positive_revision_goal)
        negative_rewritter = SyntheticRewritter(negative_seed, self.negative_revision_goal)

        # Initialize positive/negative rewriter with the sampled seeds
        if positive_seed:
            positive_rewritter.conversation_seed = positive_seed
        if negative_seed:
            negative_rewritter.conversation_seed = negative_seed

        # For both positive and negative rewriter objects do:
        for rewriter, pool in [(positive_rewritter, self.positive_pool), (negative_rewritter, self.negative_pool)]:
            tqdm.write(f"Generating revision ideas for {rewriter.revision_goal}...")
            # 3. Generate revision ideas
            rewriter.generate_revision_ideas()
            tqdm.write(f"Generated {len(rewriter.revision_ideas)} revision ideas: \n {rewriter.revision_ideas} ")

            # 4. Rewrite conversations
            tqdm.write(f"Rewriting conversations based on the revision ideas...")
            rewriter.rewrite_conversation()
            tqdm.write(f"Completed rewriting conversations following the revision ideas")


            # 5. Get conversation list
            rewrites = rewriter.get_conversation_list()
            # 6. Update the pool with current revision result
            pool.extend(rewrites)
            tqdm.write(f"Updated {rewriter.revision_goal} pool with new rewrites.")


        # iteration number increment
        self.iteration += 1
        self.positive_revisions.extend(positive_rewritter.get_conversation_list())
        self.negative_revisions.extend(negative_rewritter.get_conversation_list())
        self.positive_revision_ideas.extend(positive_rewritter.revision_ideas)
        self.negative_revision_ideas.extend(negative_rewritter.revision_ideas) 
        
    def run(self, num_of_iterations):
        for i in tqdm(range(num_of_iterations), desc="Iterative Rewriting"):
            self.single_iteration()