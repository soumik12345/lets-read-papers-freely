import json
import math
from tea_client.errors import ValidationError
from typing import List, TYPE_CHECKING

import weave
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from paperswithcode.models import Paper


class PapersWithCodeDatasetGenerator:

    def __init__(self, tasks: List[str]):
        from paperswithcode import PapersWithCodeClient

        self.tasks = tasks
        self.pwc_client = PapersWithCodeClient()

    def fetch_papers(self, task_id: str) -> List["Paper"]:
        papers_count = self.pwc_client.task_paper_list(task_id=task_id).count
        paper_results = []
        for idx in tqdm(
            range(math.ceil(papers_count / 50)),
            desc=f"Fetching papers for {task_id}",
            leave=False,
        ):
            try:
                paper_results.append(
                    self.pwc_client.task_paper_list(task_id=task_id, page=idx)
                )
            except Exception as e:
                pass
        papers = []
        for paper_results in paper_results:
            papers += paper_results.results
        return papers

    def fetch_and_publish_data(
        self, project_name: str, entity_name: str, dataset_name: str
    ):
        weave.init(project_name=f"{entity_name}/{project_name}")
        papers_list = []
        for task in tqdm(self.tasks, desc=f"Fetching data for Task"):
            papers = self.fetch_papers(task_id=task)
            pbar = tqdm(papers, leave=False, desc=f"Fetching data for {task}")
            successful_counts = 0
            for paper in pbar:
                try:
                    repository_count = self.pwc_client.paper_repository_list(
                        paper_id=paper.id
                    ).count
                except ValidationError:
                    repository_count = 0
                try:
                    method_count = self.pwc_client.paper_method_list(
                        paper_id=paper.id
                    ).count
                except ValidationError:
                    method_count = 0
                try:
                    result_count = self.pwc_client.paper_result_list(
                        paper_id=paper.id
                    ).count
                except ValidationError:
                    result_count = 0
                if result_count > 0 and method_count > 0:
                    successful_counts += 1
                    pbar.set_description(
                        f"Fetching data for {task} ({successful_counts})"
                    )
                    repositories = [
                        repo_result.url
                        for repo_result in self.pwc_client.paper_repository_list(
                            paper_id=paper.id, items_per_page=repository_count
                        ).results
                    ]
                    methods = [
                        json.loads(method_result.model_dump_json())
                        for method_result in self.pwc_client.paper_method_list(
                            paper_id=paper.id, items_per_page=method_count
                        ).results
                    ]
                    results = [
                        json.loads(result.model_dump_json())
                        for result in self.pwc_client.paper_result_list(
                            paper_id=paper.id, items_per_page=result_count
                        ).results
                    ]
                    papers_list.append(
                        {
                            "id": paper.id,
                            "url_abs": paper.url_abs,
                            "url_pdf": paper.url_pdf,
                            "title": paper.title,
                            "abstract": paper.abstract,
                            "authors": paper.authors,
                            "task": task,
                            "repository": repositories,
                            "method": methods,
                            "results": results,
                        }
                    )
            break
        weave.publish(weave.Dataset(rows=papers_list, name=dataset_name))
