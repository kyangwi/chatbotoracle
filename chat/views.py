import os
import json
import uuid
import datetime
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .utils import (
    sessions_collection,
    messages_collection,
    charts_collection,
    feedback_collection,
    get_dataset_intro_payload,
    get_session_messages_sorted,
    build_langchain_history,
    init_database,
    fetch_dataframe,
    should_generate_chart,
    has_visual_intent,
    build_chart_with_tools,
    warm_dataset_intro_cache
)
from .agent_graph import run_agent_graph

# ---------------------------------------------------------------------------
# Auth Views
# ---------------------------------------------------------------------------

def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    return render(request, 'chat/login.html', {'form': form})


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{error}")
    else:
        form = UserCreationForm()
    return render(request, 'chat/signup.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')


# ---------------------------------------------------------------------------
# Main Chat Views (login_required + user-scoped)
# ---------------------------------------------------------------------------

@login_required
def index(request):
    payload = get_dataset_intro_payload()
    context = {
        'dataset_intro_overview': payload.get('text'),
        'dataset_intro_analysis': payload.get('analysis'),
        'dataset_intro_suggestions': payload.get('suggestions', []),
        'username': request.user.username,
    }
    return render(request, 'chat/index.html', context)

@login_required
def dataset_intro(request):
    return JsonResponse(get_dataset_intro_payload())

@login_required
@csrf_exempt
def new_chat(request):
    if request.method == 'POST':
        session_id = str(uuid.uuid4())
        title = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        timestamp = datetime.datetime.now().isoformat()
        user_id = str(request.user.id)

        sessions_collection.add(
            ids=[session_id],
            metadatas=[{"title": title, "timestamp": timestamp, "user_id": user_id}],
            documents=[title]
        )

        return JsonResponse({"session_id": session_id, "title": title, "messages": []})
    return JsonResponse({"error": "Method not allowed"}, status=405)

@login_required
def get_sessions(request):
    user_id = str(request.user.id)
    try:
        results = sessions_collection.get(where={"user_id": user_id})
    except Exception:
        # Fallback: get all and filter (for legacy sessions without user_id)
        results = sessions_collection.get()

    sessions = []
    if results['ids']:
        for i, sid in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            # Only return sessions belonging to this user (handles legacy data)
            if metadata.get('user_id') and metadata.get('user_id') != user_id:
                continue
            sessions.append({
                "id": sid,
                "title": metadata.get('title', 'Untitled Chat'),
                "timestamp": metadata.get('timestamp', '')
            })

    sessions.sort(key=lambda x: x['timestamp'], reverse=True)
    return JsonResponse({"sessions": sessions})

@login_required
def get_chat(request, session_id):
    # Ownership check
    session_results = sessions_collection.get(ids=[session_id])
    if not session_results['ids']:
        return JsonResponse({"error": "Session not found"}, status=404)
    session_meta = session_results['metadatas'][0]
    if session_meta.get('user_id') and session_meta.get('user_id') != str(request.user.id):
        return JsonResponse({"error": "Forbidden"}, status=403)

    messages_data = get_session_messages_sorted(session_id)

    # Enrich with charts and feedback
    if messages_data:
        msg_ids = [m["id"] for m in messages_data if m["role"] == "bot"]
        if msg_ids:
            try:
                chart_results = charts_collection.get(ids=msg_ids)
                chart_map = {}
                if chart_results and chart_results.get("ids"):
                    for i, cid in enumerate(chart_results["ids"]):
                        try:
                            chart_map[cid] = json.loads(chart_results["documents"][i])
                        except Exception:
                            pass

                feedback_results = feedback_collection.get(ids=msg_ids)
                feedback_map = {}
                if feedback_results and feedback_results.get("ids"):
                    for i, fid in enumerate(feedback_results["ids"]):
                        meta = feedback_results["metadatas"][i] or {}
                        feedback_map[fid] = meta.get("rating")

                for msg in messages_data:
                    if msg["role"] == "bot":
                        if msg["id"] in chart_map:
                            msg["chart"] = chart_map[msg["id"]]
                        if msg["id"] in feedback_map:
                            msg["feedback"] = feedback_map[msg["id"]]
            except Exception as e:
                print(f"Error enriching messages: {e}")

    return JsonResponse({"messages": messages_data})

@login_required
@csrf_exempt
def delete_chat(request, session_id):
    if request.method == 'DELETE':
        try:
            # Ownership check
            session_results = sessions_collection.get(ids=[session_id])
            if session_results['ids']:
                session_meta = session_results['metadatas'][0]
                if session_meta.get('user_id') and session_meta.get('user_id') != str(request.user.id):
                    return JsonResponse({"error": "Forbidden"}, status=403)

            sessions_collection.delete(ids=[session_id])

            # Delete associated messages
            msg_results = messages_collection.get(where={"session_id": session_id})
            if msg_results['ids']:
                messages_collection.delete(ids=msg_results['ids'])

                # Delete associated charts & feedback
                try:
                    charts_collection.delete(ids=msg_results['ids'])
                except Exception:
                    pass
                try:
                    feedback_collection.delete(ids=msg_results['ids'])
                except Exception:
                    pass

            return JsonResponse({"status": "success"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Method not allowed"}, status=405)

@login_required
@csrf_exempt
def send_message(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message')
            session_id = data.get('session_id')

            if not user_message or not session_id:
                return JsonResponse({"error": "Missing message or session_id"}, status=400)

            # Verify session exists and belongs to user
            session_results = sessions_collection.get(ids=[session_id])
            if not session_results['ids']:
                return JsonResponse({"error": "Session not found"}, status=404)
            session_meta = session_results['metadatas'][0]
            if session_meta.get('user_id') and session_meta.get('user_id') != str(request.user.id):
                return JsonResponse({"error": "Forbidden"}, status=403)

            # Update session title if it's the first message
            msgs = get_session_messages_sorted(session_id)
            if not msgs:
                new_title = user_message[:30] + "..." if len(user_message) > 30 else user_message
                meta = session_results['metadatas'][0]
                meta['title'] = new_title
                sessions_collection.update(
                    ids=[session_id],
                    metadatas=[meta],
                    documents=[new_title]
                )

            chat_history = build_langchain_history(msgs)

            db = init_database(
                os.getenv("DB_USER"),
                os.getenv("DB_PASSWORD"),
                os.getenv("DB_NAME")
            )

            timestamp = datetime.datetime.now().isoformat()

            # --- LangGraph agentic pipeline ---
            agent_result = run_agent_graph(user_message, db, chat_history)
            response_text = agent_result.response_text
            sql_query = agent_result.sql_query
            chart_payload = agent_result.chart_payload

            # If the graph didn't produce a chart but the user wants one,
            # fall back to checking previous SQL from conversation history
            if not chart_payload and not sql_query and has_visual_intent(user_message):
                for msg in reversed(msgs):
                    prev_sql = (msg.get("metadata") or {}).get("sql_query")
                    if prev_sql:
                        sql_query = prev_sql
                        df = fetch_dataframe(db, prev_sql)
                        if df is not None and not df.empty:
                            chart_payload = build_chart_with_tools(user_message, df, prev_sql)
                        break

            user_msg_id = str(uuid.uuid4())
            messages_collection.add(
                ids=[user_msg_id],
                metadatas=[{"session_id": session_id, "role": "user", "timestamp": timestamp}],
                documents=[user_message]
            )

            resp_timestamp = datetime.datetime.now().isoformat()
            bot_msg_id = str(uuid.uuid4())
            bot_metadata = {"session_id": session_id, "role": "bot", "timestamp": resp_timestamp}
            if sql_query:
                bot_metadata["sql_query"] = sql_query
            messages_collection.add(
                ids=[bot_msg_id],
                metadatas=[bot_metadata],
                documents=[response_text]
            )

            if chart_payload:
                charts_collection.add(
                    ids=[bot_msg_id],
                    metadatas=[{"session_id": session_id}],
                    documents=[json.dumps(chart_payload)]
                )

            return JsonResponse({
                "response": [response_text],
                "chart": chart_payload,
                "user_message_id": user_msg_id,
                "bot_message_id": bot_msg_id
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed"}, status=405)


@login_required
@csrf_exempt
def regenerate_message(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get("session_id")
            target_bot_message_id = data.get("message_id")

            if not session_id:
                return JsonResponse({"error": "No session ID provided"}, status=400)

            session_results = sessions_collection.get(ids=[session_id])
            if not session_results["ids"]:
                return JsonResponse({"error": "Session not found"}, status=404)
            session_meta = session_results['metadatas'][0]
            if session_meta.get('user_id') and session_meta.get('user_id') != str(request.user.id):
                return JsonResponse({"error": "Forbidden"}, status=403)

            msgs = get_session_messages_sorted(session_id)
            if not msgs:
                return JsonResponse({"error": "No messages found for this session"}, status=404)

            latest_bot_idx = None
            for idx in range(len(msgs) - 1, -1, -1):
                if msgs[idx]["role"] == "bot":
                    latest_bot_idx = idx
                    break
            if latest_bot_idx is None:
                return JsonResponse({"error": "No bot response found to regenerate"}, status=404)

            target_bot_idx = latest_bot_idx
            found_requested_target = not bool(target_bot_message_id)
            if target_bot_message_id:
                for idx, msg in enumerate(msgs):
                    if msg["id"] == target_bot_message_id and msg["role"] == "bot":
                        target_bot_idx = idx
                        found_requested_target = True
                        break
            if not found_requested_target:
                return JsonResponse({"error": "Requested bot message was not found"}, status=404)
            if target_bot_idx != latest_bot_idx:
                return JsonResponse({"error": "Only the latest bot response can be regenerated"}, status=400)

            user_idx = None
            for idx in range(target_bot_idx - 1, -1, -1):
                if msgs[idx]["role"] == "user":
                    user_idx = idx
                    break
            if user_idx is None:
                return JsonResponse({"error": "No matching user prompt found for regeneration"}, status=400)

            user_query = msgs[user_idx]["content"]
            prior_history = build_langchain_history(msgs[:user_idx])

            db = init_database(
                os.getenv("DB_USER"), os.getenv("DB_PASSWORD"), os.getenv("DB_NAME")
            )

            # --- LangGraph agentic pipeline (regenerate) ---
            agent_result = run_agent_graph(user_query, db, prior_history)
            response_text = agent_result.response_text
            sql_query = agent_result.sql_query
            chart_payload = agent_result.chart_payload

            bot_message = msgs[target_bot_idx]
            bot_msg_id = bot_message["id"]
            bot_metadata = dict(bot_message.get("metadata") or {})
            bot_metadata["timestamp"] = datetime.datetime.now().isoformat()

            messages_collection.update(
                ids=[bot_msg_id],
                metadatas=[bot_metadata],
                documents=[response_text],
            )

            if chart_payload:
                charts_collection.upsert(
                    ids=[bot_msg_id],
                    metadatas=[{"session_id": session_id}],
                    documents=[json.dumps(chart_payload)],
                )
            else:
                try:
                    charts_collection.delete(ids=[bot_msg_id])
                except Exception:
                    pass

            try:
                feedback_collection.delete(ids=[bot_msg_id])
            except Exception:
                pass

            return JsonResponse({
                "response": [response_text],
                "chart": chart_payload,
                "bot_message_id": bot_msg_id,
                "regenerated": True,
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed"}, status=405)

@login_required
@csrf_exempt
def message_feedback(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            session_id = data.get("session_id")
            message_id = data.get("message_id")
            rating = (data.get("rating") or "").strip().lower()

            if not session_id or not message_id:
                return JsonResponse({"error": "session_id and message_id are required"}, status=400)

            if rating not in ["up", "down"]:
                return JsonResponse({"error": "rating must be 'up' or 'down'"}, status=400)

            updated_at = datetime.datetime.now().isoformat()
            feedback_collection.upsert(
                ids=[message_id],
                metadatas=[{
                    "session_id": session_id,
                    "rating": rating,
                    "updated_at": updated_at,
                }],
                documents=["Message feedback"],
            )

            return JsonResponse({"status": "success", "message_id": message_id, "rating": rating})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Method not allowed"}, status=405)

# Initialize dataset cache on module load
warm_dataset_intro_cache()
