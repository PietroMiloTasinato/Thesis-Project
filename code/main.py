import argparse
from tqdm import tqdm

from utils import data_io, s2
from knowledge.store import KnowledgeStore
#from models.openai import OpenAIClient
from models.CustomAgent import OllamaClient, HOST
from pipelines.research_pipeline import ResearchPipeline
from pipelines.review_pipeline import ReviewPipeline
from utils import jutil

def fetch_resources(paper: dict, knowledge_store: KnowledgeStore):
    references = s2.get_relevant_references(paper)
    entities = knowledge_store.get_relevant_entities(
        [paper['corpusId']] + [reference['corpusId'] for reference in references]
    )
    return references, entities


def run(
    paper_ids: list,
    knowledge_store: KnowledgeStore,
    client: OllamaClient
):
    
    print("PAPERS | SETTING UP PAPERS")

    papers = s2.filter_papers(
        s2.get_papers(paper_ids),
        categories=['title', 'abstract', 'embedding']
    )

    results = []

    print("PAPERS | FINSCED SETTING UP PAPERS")

    for paper in tqdm(papers):
        context = {'paper': {key: paper.get(key) for key in ('title', 'abstract')}}
        
        references, entities = fetch_resources(paper, knowledge_store)
        context.update(references=references, entities=entities)

        print("RESEARCH PIPELINE | LAUNCH")

        research_pipeline = ResearchPipeline(api_client=client)
        context = research_pipeline.run(context)

        results.append(context)
        data_io.save_result('./results/ideas.jsonl', context)

    return results

def reviewer_run(paper_files : list, client: OllamaClient):
    
    results = []
    
    #print("PAPERS | FINSCED SETTING UP PAPERS")

    for paper_entry in tqdm(papers_files):
        
        context = jutil.json_to_sections(f"p{paper_entry}.json")
        
        #context = {'paper': {key: paper.get(key) for key in ('title', 'abstract')}}
        
        #references, entities = fetch_resources(paper, knowledge_store)
        #context.update(references=references, entities=entities)

        print("REVIEW PIPELINE | LAUNCH")

        review_pipeline = ReviewPipeline(api_client=client)
        context = review_pipeline.run(context)

        results.append(context)
        data_io.save_result('./results/ideas.jsonl', context)
    
    return results

if __name__ == "__main__":
    print("MAIN | initializing")
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data-path', '-d', default='./data/papers.jsonl')
    argparser.add_argument('--knowledge-path', '-k', default='./data/knowledge.jsonl')
    argparser.add_argument('--agent-type', '-a', default='reviewer')
    args = argparser.parse_args()

    if(args.agent_type == "reviewer"):
        
        papers_files = []
        
        for line in open("./Local_papers/index.txt"):
            papers_files.append(line[:-1])

        #print(papers_files)

        client = OllamaClient(host=HOST)
        #print("MAIN | running")
        results = reviewer_run(papers_files, client)
    
    else:
        paper_ids = data_io.load_paper_ids(args.data_path)
        knowledge_store = KnowledgeStore(args.knowledge_path)

        client = OllamaClient(host=HOST)
        #print("MAIN | running")
        results = run(paper_ids, knowledge_store, client)
