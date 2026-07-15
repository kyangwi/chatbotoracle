# LangGraph Chatbot State Machine Structure

This is the visual structure of the compiled LangGraph StateGraph, saved locally in the project repository.

## 1. Visual Graph Diagram (PNG)

Here is the graph image generated directly from the LangGraph runtime:

![Compiled LangGraph State Machine](chat_graph.png)

---

## 2. Mermaid Structure Definition

You can also render the diagram using this Mermaid block:

```mermaid
graph TD
    %% Nodes
    classify_intent["classify_intent"]
    generate_sql["generate_sql"]
    execute_sql["execute_sql"]
    fix_sql["fix_sql"]
    handle_sql_error["handle_sql_error"]
    generate_response["generate_response"]
    handle_data_overview["handle_data_overview"]
    handle_no_sql["handle_no_sql"]
    maybe_generate_chart["maybe_generate_chart"]
    finalize["finalize"]
    END["END"]

    %% Edges
    classify_intent -->|data_overview| handle_data_overview
    classify_intent -->|no_sql| handle_no_sql
    classify_intent -->|sql_query| generate_sql

    generate_sql -->|NO_SQL| handle_no_sql
    generate_sql -->|sql_query| execute_sql

    execute_sql -->|success| generate_response
    execute_sql -->|error & retries < 2| fix_sql
    execute_sql -->|error & retries >= 2| handle_sql_error

    fix_sql --> execute_sql
    handle_sql_error --> finalize

    generate_response --> maybe_generate_chart
    maybe_generate_chart --> finalize

    handle_data_overview --> finalize
    handle_no_sql --> finalize

    finalize --> END
```
