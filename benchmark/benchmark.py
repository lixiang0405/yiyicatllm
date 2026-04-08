"""
推理性能 Benchmark 脚本
测试 vLLM 部署模型的吞吐量、延迟等关键指标。
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai


# ============================================
# 测试用例
# ============================================
TEST_PROMPTS = [
    "请介绍一下中国科学技术大学的历史。",
    "中科大少年班是什么？有什么特色？",
    "中科大有哪些优势学科？",
    "如何报考中科大的研究生？",
    "中科大在人工智能领域有哪些研究成果？",
    "中科大的校园生活怎么样？",
    "中科大毕业生的就业情况如何？",
    "请介绍一下中科大的量子信息研究。",
    "中科大和清华北大相比有什么优势？",
    "中科大的计算机科学专业怎么样？",
]


def single_request(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 256,
) -> dict:
    """发送单个请求并记录性能指标"""
    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是中科大智能问答助手。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    end_time = time.perf_counter()
    latency = end_time - start_time

    output_text = response.choices[0].message.content
    output_tokens = response.usage.completion_tokens
    input_tokens = response.usage.prompt_tokens

    return {
        "prompt": prompt[:50] + "...",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": round(latency, 3),
        "tokens_per_second": round(output_tokens / latency, 1),
        "output_preview": output_text[:100] + "...",
    }


def single_request_streaming(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 256,
) -> dict:
    """发送流式请求，测量 TTFT 和逐 token 延迟"""
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是中科大智能问答助手。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1

    end_time = time.perf_counter()

    total_latency = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_latency
    generation_time = end_time - first_token_time if first_token_time else total_latency

    return {
        "prompt": prompt[:50] + "...",
        "output_tokens": token_count,
        "ttft_ms": round(ttft * 1000, 1),
        "total_latency_seconds": round(total_latency, 3),
        "generation_tokens_per_second": round(token_count / generation_time, 1) if generation_time > 0 else 0,
    }


def run_throughput_test(
    client: openai.OpenAI,
    model: str,
    concurrency: int = 4,
    num_requests: int = 10,
    max_tokens: int = 256,
) -> dict:
    """并发吞吐量测试"""
    print(f"\n{'='*50}")
    print(f"  吞吐量测试 (并发={concurrency}, 请求数={num_requests})")
    print(f"{'='*50}")

    prompts = [TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(num_requests)]
    results = []
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(single_request, client, model, prompt, max_tokens): prompt
            for prompt in prompts
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"  ✓ {result['prompt']} -> {result['tokens_per_second']} tok/s")
            except Exception as error:
                print(f"  ✗ 请求失败: {error}")

    total_time = time.perf_counter() - start_time
    total_output_tokens = sum(r["output_tokens"] for r in results)

    summary = {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "successful_requests": len(results),
        "total_time_seconds": round(total_time, 2),
        "total_output_tokens": total_output_tokens,
        "overall_throughput_tokens_per_second": round(total_output_tokens / total_time, 1),
        "avg_latency_seconds": round(statistics.mean(r["latency_seconds"] for r in results), 3),
        "p50_latency_seconds": round(statistics.median(r["latency_seconds"] for r in results), 3),
        "p99_latency_seconds": round(
            sorted(r["latency_seconds"] for r in results)[int(len(results) * 0.99)], 3
        ) if results else 0,
    }

    return summary


def run_latency_test(
    client: openai.OpenAI,
    model: str,
    num_requests: int = 5,
    max_tokens: int = 256,
) -> dict:
    """单请求延迟测试（含 TTFT）"""
    print(f"\n{'='*50}")
    print(f"  延迟测试 (请求数={num_requests})")
    print(f"{'='*50}")

    results = []
    for i in range(num_requests):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        result = single_request_streaming(client, model, prompt, max_tokens)
        results.append(result)
        print(f"  ✓ TTFT={result['ttft_ms']}ms, 生成速度={result['generation_tokens_per_second']} tok/s")

    summary = {
        "num_requests": num_requests,
        "avg_ttft_ms": round(statistics.mean(r["ttft_ms"] for r in results), 1),
        "p50_ttft_ms": round(statistics.median(r["ttft_ms"] for r in results), 1),
        "avg_generation_speed_tokens_per_second": round(
            statistics.mean(r["generation_tokens_per_second"] for r in results), 1
        ),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="推理性能 Benchmark")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="outputs/ustc-qa-quantized-awq-int4")
    parser.add_argument("--concurrency", type=int, default=4, help="并发数")
    parser.add_argument("--num-requests", type=int, default=10, help="总请求数")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--output", type=str, default="benchmark/results.json", help="结果输出文件")
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.api_base, api_key="not-needed")

    print("=" * 50)
    print("  中科大智能问答助手 - 推理性能 Benchmark")
    print(f"  API: {args.api_base}")
    print(f"  模型: {args.model}")
    print("=" * 50)

    # 预热
    print("\n预热中...")
    try:
        single_request(client, args.model, "你好", max_tokens=16)
        print("  预热完成 ✓")
    except Exception as error:
        print(f"  ❌ 无法连接到推理服务: {error}")
        print("  请确认 vLLM 服务已启动: bash deploy/serve.sh")
        return

    # 延迟测试
    latency_results = run_latency_test(
        client, args.model, num_requests=5, max_tokens=args.max_tokens
    )

    # 吞吐量测试（不同并发）
    throughput_results = {}
    for concurrency in [1, 2, 4, 8]:
        if concurrency > args.concurrency:
            break
        result = run_throughput_test(
            client, args.model,
            concurrency=concurrency,
            num_requests=max(concurrency * 2, args.num_requests),
            max_tokens=args.max_tokens,
        )
        throughput_results[f"concurrency_{concurrency}"] = result

    # 汇总结果
    all_results = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "latency": latency_results,
        "throughput": throughput_results,
    }

    # 保存结果
    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"  Benchmark 结果汇总")
    print(f"{'='*60}")
    print(f"  模型: {args.model}")
    print(f"")
    print(f"  【延迟指标】")
    print(f"    平均 TTFT:       {latency_results['avg_ttft_ms']} ms")
    print(f"    P50 TTFT:        {latency_results['p50_ttft_ms']} ms")
    print(f"    平均生成速度:     {latency_results['avg_generation_speed_tokens_per_second']} tok/s")
    print(f"")
    print(f"  【吞吐量指标】")
    for key, value in throughput_results.items():
        concurrency = value["concurrency"]
        throughput = value["overall_throughput_tokens_per_second"]
        avg_latency = value["avg_latency_seconds"]
        print(f"    并发={concurrency}: {throughput} tok/s, 平均延迟={avg_latency}s")
    print(f"")
    print(f"  结果已保存至: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
