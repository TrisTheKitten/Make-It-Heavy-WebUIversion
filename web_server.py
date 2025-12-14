import re
import json
import time
import queue
import threading
import yaml
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from agent import create_agent
from orchestrator import TaskOrchestrator
from megamind_orchestrator import MegamindOrchestrator

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

progress_queues = {}
progress_lock = threading.Lock()


def strip_special_chars(text: str) -> str:
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    text = re.sub(r'[●○◐◑◒◓✓✗×·]', '', text)
    text = re.sub(r'[🔄🔧📞💭✅❌⚠️🚨📋🔍📝💡🎯⚡🚀]', '', text)
    text = re.sub(r'\*{3,}', '**', text)
    text = re.sub(r'#{4,}\s*', '### ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\*\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)


def save_config(config):
    with open("config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    config = load_config()
    safe_config = {
        'provider': config.get('provider', 'gemini'),
        'gemini': {'model': config['gemini']['model']},
        'openrouter': {'model': config['openrouter']['model']},
        'orchestrator': {
            'parallel_agents': config['orchestrator']['parallel_agents'],
            'task_timeout': config['orchestrator']['task_timeout']
        },
        'agent': {'max_iterations': config['agent']['max_iterations']}
    }
    return jsonify(safe_config)


@app.route('/api/config', methods=['POST'])
def update_config():
    try:
        updates = request.json
        config = load_config()
        
        if 'provider' in updates:
            config['provider'] = updates['provider']
        if 'parallel_agents' in updates:
            config['orchestrator']['parallel_agents'] = int(updates['parallel_agents'])
        if 'task_timeout' in updates:
            config['orchestrator']['task_timeout'] = int(updates['task_timeout'])
        if 'max_iterations' in updates:
            config['agent']['max_iterations'] = int(updates['max_iterations'])
        if 'gemini_model' in updates:
            config['gemini']['model'] = updates['gemini_model']
        if 'openrouter_model' in updates:
            config['openrouter']['model'] = updates['openrouter_model']
        
        save_config(config)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/run/single', methods=['POST'])
def run_single_agent():
    data = request.json
    query = data.get('query', '')
    session_id = data.get('session_id', 'default')
    images = data.get('images', [])
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    with progress_lock:
        progress_queues[session_id] = queue.Queue()
    
    def run_agent():
        q = progress_queues.get(session_id)
        try:
            q.put({'type': 'status', 'message': 'Initializing agent...'})
            q.put({'type': 'progress', 'agent_id': 0, 'status': 'PROCESSING'})
            
            agent = create_agent(silent=True)
            
            q.put({'type': 'status', 'message': 'Processing query...'})
            
            response = agent.run(query, images=images)
            formatted_response = strip_special_chars(response)
            
            q.put({'type': 'progress', 'agent_id': 0, 'status': 'COMPLETED'})
            q.put({'type': 'result', 'content': formatted_response})
            q.put({'type': 'done'})
            
        except Exception as e:
            q.put({'type': 'progress', 'agent_id': 0, 'status': 'FAILED'})
            q.put({'type': 'error', 'message': str(e)})
            q.put({'type': 'done'})
    
    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/run/multi', methods=['POST'])
def run_multi_agent():
    data = request.json
    query = data.get('query', '')
    session_id = data.get('session_id', 'default')
    images = data.get('images', [])
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    with progress_lock:
        progress_queues[session_id] = queue.Queue()
    
    def run_orchestrator():
        q = progress_queues.get(session_id)
        try:
            q.put({'type': 'status', 'message': 'Initializing orchestrator...'})
            
            orchestrator = TaskOrchestrator(silent=True)
            num_agents = orchestrator.num_agents
            
            q.put({'type': 'config', 'num_agents': num_agents})
            
            for i in range(num_agents):
                q.put({'type': 'progress', 'agent_id': i, 'status': 'QUEUED'})
            
            q.put({'type': 'status', 'message': 'Decomposing task into subtasks...'})
            
            def progress_monitor():
                last_status = {}
                while True:
                    try:
                        current = orchestrator.get_progress_status()
                        for agent_id, status in current.items():
                            if last_status.get(agent_id) != status:
                                q.put({'type': 'progress', 'agent_id': agent_id, 'status': status})
                                last_status[agent_id] = status
                        time.sleep(0.3)
                    except:
                        break
            
            monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
            monitor_thread.start()
            
            def status_callback(msg):
                q.put({'type': 'stage', 'stage': 'aggregating', 'message': msg})
            
            result_data = orchestrator.orchestrate(query, images=images, status_callback=status_callback)
            formatted_result = strip_special_chars(result_data['final_result'])
            agent_results = result_data['agent_results']
            
            for i in range(num_agents):
                q.put({'type': 'progress', 'agent_id': i, 'status': 'COMPLETED'})
            
            # Send thinking process
            formatted_agent_results = []
            for ar in agent_results:
                formatted_ar = ar.copy()
                if 'response' in formatted_ar:
                    formatted_ar['response'] = strip_special_chars(formatted_ar['response'])
                formatted_agent_results.append(formatted_ar)
                
            q.put({'type': 'thinking', 'content': formatted_agent_results})
            q.put({'type': 'result', 'content': formatted_result})
            q.put({'type': 'done'})
            
        except Exception as e:
            q.put({'type': 'error', 'message': str(e)})
            q.put({'type': 'done'})
    
    thread = threading.Thread(target=run_orchestrator, daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/run/megamind', methods=['POST'])
def run_megamind():
    data = request.json
    query = data.get('query', '')
    session_id = data.get('session_id', 'default')
    images = data.get('images', [])
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    with progress_lock:
        progress_queues[session_id] = queue.Queue()
    
    def run_megamind_orchestrator():
        q = progress_queues.get(session_id)
        try:
            q.put({'type': 'status', 'message': 'Initializing Megamind orchestrator...'})
            q.put({'type': 'megamind_config', 'stages': MegamindOrchestrator.STAGES})
            
            orchestrator = MegamindOrchestrator(silent=True)
            
            def progress_monitor():
                last_status = {}
                while True:
                    try:
                        current = orchestrator.get_progress_status()
                        for stage, status in current.get('stages', {}).items():
                            if last_status.get(f'stage_{stage}') != status:
                                q.put({'type': 'megamind_stage', 'stage': stage, 'status': status})
                                last_status[f'stage_{stage}'] = status
                        for agent_id, info in current.get('agents', {}).items():
                            agent_status = info.get('status') if isinstance(info, dict) else info
                            if last_status.get(f'agent_{agent_id}') != agent_status:
                                q.put({'type': 'megamind_agent', 'agent_id': agent_id, 'status': agent_status})
                                last_status[f'agent_{agent_id}'] = agent_status
                        time.sleep(0.3)
                    except:
                        break
            
            monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
            monitor_thread.start()
            
            def status_callback(msg):
                q.put({'type': 'status', 'message': msg})
            
            result_data = orchestrator.orchestrate(query, images=images, status_callback=status_callback)
            
            formatted_result = strip_special_chars(result_data['final_result'])
            
            thinking_data = {
                'questions': result_data.get('questions', []),
                'research_results': [],
                'first_draft': strip_special_chars(result_data.get('first_draft', '')),
                'validation_results': []
            }
            
            for r in result_data.get('research_results', []):
                thinking_data['research_results'].append({
                    'agent_id': r.get('agent_id'),
                    'response': strip_special_chars(r.get('response', ''))
                })
            
            for r in result_data.get('validation_results', []):
                thinking_data['validation_results'].append({
                    'agent_id': r.get('agent_id'),
                    'response': strip_special_chars(r.get('response', ''))
                })
            
            q.put({'type': 'megamind_thinking', 'content': thinking_data})
            q.put({'type': 'result', 'content': formatted_result})
            q.put({'type': 'done'})
            
        except Exception as e:
            q.put({'type': 'error', 'message': str(e)})
            q.put({'type': 'done'})
    
    thread = threading.Thread(target=run_megamind_orchestrator, daemon=True)
    thread.start()
    
    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/stream/<session_id>')
def stream_progress(session_id):
    def generate():
        q = progress_queues.get(session_id)
        if not q:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
            return
        
        start_time = time.time()
        
        while True:
            try:
                try:
                    data = q.get(timeout=0.5)
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    if data.get('type') == 'done':
                        break
                except queue.Empty:
                    elapsed = int(time.time() - start_time)
                    yield f"data: {json.dumps({'type': 'heartbeat', 'elapsed': elapsed})}\n\n"
                    
            except GeneratorExit:
                break
        
        with progress_lock:
            if session_id in progress_queues:
                del progress_queues[session_id]
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


if __name__ == '__main__':
    print("Starting Make It Heavy Web Server...")
    print("Open http://localhost:5050 in your browser")
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
