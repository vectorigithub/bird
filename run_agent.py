from train_agent import regex_search, generate_example

def main():
    while True:
        try:
            q = input("\n[you] > ")
        except EOFError:
            break

        if q.strip().lower() in {"exit", "quit"}:
            break

        if not q.strip():
            continue

        pattern = q  # you can map natural language → regex later
        snippets = regex_search(pattern)
        context = "\n\n".join(snippets[:3])

        prompt = f"Context:\n{context}\n\nTask:\n{q}\n\nAnswer:\n"
        out = generate_example(prompt, max_new_tokens=256)
        print("\n[agent] >")
        print(out)

if __name__ == "__main__":
    main()
