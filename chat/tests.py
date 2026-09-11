import os
from django.test import TestCase
from chat.utils import (
    is_sql_query_question,
    is_data_overview_question,
    has_table_intent,
    has_visual_intent,
    get_targeted_glossary_context,
    get_exchange_rates_context,
    extract_clean_sql,
    normalize_response_text,
)
from chat.financial_tools import (
    financial_calculator,
    calculate_rate_of_change,
    calculate_cagr,
    parse_and_compute_calculation_request,
)
from chat.agent_graph import classify_intent, IntentType, _compiled_graph, route_by_intent


class ChatPipelineTests(TestCase):
    def test_intent_heuristics_fast_path(self):
        # SQL question patterns
        self.assertTrue(is_sql_query_question("What is the total revenue for September 2026?"))
        self.assertTrue(is_sql_query_question("show top 10 cost centers by total expenditure in 2021"))
        self.assertTrue(is_sql_query_question("how many transactions were posted on 9th sep 2026"))
        self.assertTrue(is_sql_query_question("check the database for voucher payments"))

        # Data overview patterns
        self.assertTrue(is_data_overview_question("What is this data all about?"))
        self.assertTrue(is_data_overview_question("give me an overview of the dataset"))
        self.assertFalse(is_sql_query_question("What is this data all about?"))

        # Table & Visual intent
        self.assertTrue(has_table_intent("show this as a table"))
        self.assertTrue(has_table_intent("tabulate monthly revenues"))
        self.assertTrue(has_visual_intent("generate a bar chart of top 10 cost centers"))
        self.assertTrue(has_visual_intent("plot the trend as a line graph"))

    def test_targeted_glossary_optimization(self):
        # Empty or general question gets compact primer
        primer = get_targeted_glossary_context("")
        self.assertTrue(len(primer) < 300)
        self.assertIn("UGX", primer)

        # Question mentioning specific term gets targeted definition
        efris_context = get_targeted_glossary_context("What is the EFRIS compliance status?")
        self.assertIn("EFRIS", efris_context)
        self.assertTrue(len(efris_context) < 1000)

        # Question without forex terms skips exchange rates
        no_fx = get_exchange_rates_context("What is total expenditure in 2021?")
        self.assertEqual(no_fx, "")

        # Question with USD/forex gets rates
        fx_rates = get_exchange_rates_context("Convert this to USD exchange rate")
        # Rates string is either loaded or empty on network timeout, but shouldn't error
        self.assertIsInstance(fx_rates, str)

    def test_financial_tools(self):
        # Safe math
        calc = financial_calculator.invoke({"expression": "(150000000 - 100000000) / 100000000 * 100"})
        self.assertIn("50", calc)

        # Rate of change
        roc = calculate_rate_of_change.invoke({"old_value": 100.0, "new_value": 150.0, "label": "Revenue"})
        self.assertIn("+50.00%", roc)
        self.assertIn("Increase", roc)

        # CAGR
        cagr = calculate_cagr.invoke({"start_value": 1000.0, "end_value": 1331.0, "periods": 3, "label": "Growth"})
        self.assertIn("10.00%", cagr)

        # Direct natural language calculation parser
        auto_calc = parse_and_compute_calculation_request("rate of change from 200 to 300")
        self.assertIsNotNone(auto_calc)
        self.assertIn("50.00%", auto_calc)

    def test_extract_clean_sql(self):
        raw_md = "```sql\nSELECT * FROM staging.proddta_f0911_account_ledger LIMIT 10;\n```"
        self.assertEqual(extract_clean_sql(raw_md), "SELECT * FROM staging.proddta_f0911_account_ledger LIMIT 10;")

        raw_commented = "-- Query description\nSELECT COUNT(*) FROM staging.proddta_f0911_account_ledger WHERE glpost = 'P';"
        self.assertEqual(extract_clean_sql(raw_commented), "SELECT COUNT(*) FROM staging.proddta_f0911_account_ledger WHERE glpost = 'P';")

    def test_classify_intent_node(self):
        # Pure greeting
        res_hi = classify_intent({"user_query": "Hello", "chat_history": []})
        self.assertEqual(res_hi["intent"], IntentType.NO_SQL)
        self.assertEqual(route_by_intent(res_hi), "handle_no_sql")

        # Data overview
        res_ov = classify_intent({"user_query": "What is this data all about?", "chat_history": []})
        self.assertEqual(res_ov["intent"], IntentType.DATA_OVERVIEW)
        self.assertEqual(route_by_intent(res_ov), "handle_data_overview")

        # Direct SQL question
        res_sql = classify_intent({"user_query": "What was the total expenditure in 2021?", "chat_history": []})
        self.assertEqual(res_sql["intent"], IntentType.SQL_QUERY)
        self.assertEqual(route_by_intent(res_sql), "generate_sql")

        # Financial calculation
        res_calc = classify_intent({"user_query": "Calculate rate of change from 500M to 750M", "chat_history": []})
        self.assertEqual(res_calc["intent"], IntentType.FINANCIAL_CALCULATION)
        self.assertEqual(route_by_intent(res_calc), "finalize")

    def test_compiled_graph_structure(self):
        # Verify compiled LangGraph instance has valid nodes and entrypoint
        self.assertIsNotNone(_compiled_graph)
        nodes = list(_compiled_graph.nodes.keys())
        expected_nodes = [
            "classify_intent", "generate_sql", "execute_sql", "fix_sql",
            "handle_sql_error", "generate_response", "handle_data_overview",
            "handle_no_sql", "maybe_generate_chart", "finalize"
        ]
        for node in expected_nodes:
            self.assertIn(node, nodes)
