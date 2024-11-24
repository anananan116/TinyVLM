from .modeling_llama import LlamaForCausalLM
from .modeling_VLM import AtriVLM
from .configuration_llama import VLMConfig
from .visual_modeling import CLIPModel
from .configuration_clip import CLIPConfig
from transformers import AutoTokenizer, AutoProcessor

try:
    from .creditials import hugging_face_token
except:
    hugging_face_token = None

modified_chat_template ="""{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}
        {%- if message.role == 'assistant' %}
            {% generation %}
            {{- message['content'] | trim }}
            {% endgeneration %}
        {%- else %}
            {{- message['content'] | trim }}
        {%- endif %}
        {{- '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {% generation %}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
            {% endgeneration %}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {% generation %}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
            {% endgeneration %}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
    {% generation %}
    {% endgeneration %}
{%- endif %}"""


def get_model_and_tokenizer(model_args, additional_tokens_dict, device="cuda", load_vision_model=False):
    pretrained_model = 'meta-llama/Llama-3.2-1B-Instruct' if 'pretrained_model' not in model_args else model_args['pretrained_model']
    
    config = VLMConfig.from_pretrained(pretrained_model, **model_args)
    config.load_vision_model = load_vision_model
    config.pretrained_model = pretrained_model
    if load_vision_model:
        config.visual_config = CLIPConfig.from_pretrained(config.pretrained_vision_model)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, token = hugging_face_token, config = {"chat_template": modified_chat_template})
    tokenizer.add_tokens([v for k,v in additional_tokens_dict.items()], special_tokens=True)
    special_token_map = {k: (v, tokenizer.convert_tokens_to_ids(v)) for k,v in additional_tokens_dict.items()}
    new_tokenizer_len = len(tokenizer)
    config.adjust_embedding_len = new_tokenizer_len
    config.special_token_map = special_token_map
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model = AtriVLM.from_pretrained(pretrained_model, config = config)
    if config.load_vision_model and model.visual is None:
        model.visual = CLIPModel.from_pretrained(config.pretrained_vision_model)
    
    if config.adjust_embedding_len:
        model.resize_token_embeddings(config.adjust_embedding_len, mean_resizing=True)
    processor = AutoProcessor.from_pretrained(config.pretrained_vision_model).image_processor
    if config.lora:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, target_modules=config.lora_modules, task_type=config.task_type)
        model = get_peft_model(model, lora_config)
    
    return model, tokenizer, special_token_map, processor