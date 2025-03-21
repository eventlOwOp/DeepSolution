from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncGenerator, List
from contextlib import asynccontextmanager
import uvicorn
import os
import json
import asyncio
from ours_framework import GameTreeRAG
from utils.openai_api import OpenaiAPI
from rag import HippoRAGClient

# Global variables
rag_client = None  # 共享的RAG客户端
llm_client = None  # 共享的LLM客户端
active_tasks = {}  # 活跃任务状态
task_results = {}  # 任务结果
task_counter = 0  # 任务计数器


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """初始化和清理资源的生命周期管理器"""
    global rag_client, llm_client

    # 初始化共享的RAG客户端
    rag_client = HippoRAGClient(
        save_dir="../HippoRAG/outputs",
        llm_model_name="gpt-4o-mini",
        llm_base_url="https://xiaoai.plus/v1",
        embedding_model_name="text-embedding-3-large",
        llm_api_key="***",
    )

    # 创建LLM客户端
    llm_client = OpenaiAPI(
        cache_dir="./cache",
        base_url="https://xiaoai.plus/v1",
        api_key="***",
        model_name="gpt-4o-mini",
    )

    print("RAG和LLM客户端已初始化")
    yield
    # 清理资源
    print("RAG和LLM客户端已关闭")


app = FastAPI(lifespan=lifespan)


class QueryInput(BaseModel):
    query: str


class TaskStatus(BaseModel):
    task_id: int
    status: str
    progress: Optional[float] = None
    result: Optional[Dict] = None


def create_gametree_rag():
    """为每个任务创建一个新的GameTreeRAG实例"""
    global rag_client, llm_client

    return GameTreeRAG(
        llm=llm_client,
        rag=rag_client,
        max_depth=3,
        children_num=2,
        layer_top_k=1,
        retrieval_top_k=10,
        doubt_max_new_tokens=2048,
        solution_max_new_tokens=2048,
        if_only_reference="semionly",
        if_sum_reference="sum",
        if_rerank="llmrerank",
    )


async def process_query_task_async(task_id: int, query: str):
    """异步任务处理查询"""
    global active_tasks, task_results

    try:
        # 为此任务创建一个新的GameTreeRAG实例
        rag_instance = create_gametree_rag()

        # 将实例存储在任务信息中，以便可以访问其节点记录
        active_tasks[task_id] = {
            "status": "running",
            "progress": 0.0,
            "rag_instance": rag_instance,
        }

        # 定期更新进度的任务
        async def update_progress():
            while (
                task_id in active_tasks and active_tasks[task_id]["status"] == "running"
            ):
                active_tasks[task_id]["progress"] = rag_instance.progress
                await asyncio.sleep(0.5)

        # 启动进度更新任务
        progress_task = asyncio.create_task(update_progress())

        # 运行异步解决方案生成
        solution, nodes_record = await rag_instance.get_final_solution_async(query)

        # 取消进度更新任务
        progress_task.cancel()

        # 存储结果
        task_results[task_id] = {"solution": solution, "nodes_record": nodes_record}
        active_tasks[task_id] = {"status": "completed", "progress": 1.0}

    except Exception as e:
        active_tasks[task_id] = {"status": "failed", "error": str(e)}
        print(f"Task {task_id} failed: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def get_visualization(request: Request):
    """提供树形可视化HTML页面"""
    try:
        with open("tree_visual.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error loading visualization: {str(e)}</h1>", status_code=500
        )


@app.post("/get_solution", response_model=TaskStatus)
async def get_solution(query_input: QueryInput, background_tasks: BackgroundTasks):
    """启动异步解决方案生成任务"""
    global task_counter, active_tasks

    if rag_client is None or llm_client is None:
        raise HTTPException(status_code=503, detail="RAG系统尚未初始化")

    # 创建新任务
    task_id = task_counter
    task_counter += 1

    # 在后台启动异步任务
    background_tasks.add_task(process_query_task_async, task_id, query_input.query)

    # 返回任务ID
    return TaskStatus(task_id=task_id, status="started", progress=0.0)


@app.get("/task/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: int):
    """获取任务状态和节点记录"""
    global active_tasks, task_results

    task_id = int(task_id)

    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task_info = active_tasks[task_id]
    result = {}

    if task_info["status"] == "completed" and task_id in task_results:
        # 对于已完成的任务，返回完整结果
        result = task_results[task_id]
        return TaskStatus(
            task_id=task_id,
            status="completed",
            progress=1.0,
            result=result,
        )
    elif task_info["status"] == "failed":
        # 对于失败的任务，返回错误信息
        return TaskStatus(
            task_id=task_id,
            status="failed",
            progress=0.0,
            result={"error": task_info.get("error", "Unknown error")},
        )
    else:
        # 对于正在运行的任务，获取当前的节点记录（如果有）
        rag_instance = task_info.get("rag_instance")
        if rag_instance and hasattr(rag_instance, "all_nodes_record_dict"):
            result = {"nodes_record": rag_instance.all_nodes_record_dict}

        return TaskStatus(
            task_id=task_id,
            status=task_info["status"],
            progress=task_info.get("progress", 0.0),
            result=result,
        )


@app.get("/get_all_tasks")
async def get_all_tasks():
    """获取所有任务的状态"""
    global active_tasks, task_results

    tasks = []
    for task_id, task_info in active_tasks.items():
        task_data = {
            "task_id": task_id,
            "status": task_info["status"],
            "progress": task_info.get("progress", 0.0),
        }

        if task_info["status"] == "completed" and task_id in task_results:
            task_data["query"] = (
                task_results[task_id]
                .get("nodes_record", {})
                .get("0", {})
                .get("question", "Unknown query")
            )

        tasks.append(task_data)

    return {"tasks": tasks}


def main():
    """启动服务器"""
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Error starting server: {str(e)}")


if __name__ == "__main__":
    main()
