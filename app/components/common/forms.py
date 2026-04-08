import streamlit as st

from app.utils.formatter import field_formater

# def render_raw_input_form(form_key: str, fields: dict, columns: int = 1, submit_label: str = "Run Prediction"):
#     with st.form(form_key):
#         payload = {}
#         field_items = list(fields.items())
#         fields_per_column = len(field_items) // columns + (1 if len(field_items) % columns else 0)
        
#         cols = st.columns(columns)

#         for field_name, field_spec in fields.items():
#             field_type = field_spec["type"]
#             extra = field_spec.get("extra", None)
#             if field_type == "select":
#                 options = field_spec["options"]
#                 payload[field_name] = st.selectbox(
#                     field_formater(field_name, extra),
#                     options,
#                     index=options.index(field_spec["default"]) if field_spec["default"] in options else 0,
#                     key=f"{form_key}_{field_name}",
#                 )
#             else:
#                 payload[field_name] = st.number_input(
#                     field_formater(field_name, extra),
#                     min_value=field_spec.get("min", 0),
#                     max_value=field_spec.get("max"),
#                     value=field_spec.get("default", 0),
#                     step=field_spec.get("step", 1),
#                     key=f"{form_key}_{field_name}",
#                 )
#         submitted = st.form_submit_button("Run Prediction")
#     return submitted, payload

def render_raw_input_form(form_key: str, fields: dict, columns: int = 1, submit_label: str = "Run Prediction"):
    """
    Render a dynamic form with customizable columns
    
    Args:
        form_key: Unique key for the form
        fields: Dictionary of field specifications
        columns: Number of columns to arrange fields (default: 1)
        submit_label: Label for submit button
        
    Returns:
        Tuple of (submitted, payload)
    """
    with st.form(form_key):
        payload = {}
        
        # Convert fields to list for column distribution
        field_items = list(fields.items())
        
        # Calculate fields per column
        fields_per_column = len(field_items) // columns + (1 if len(field_items) % columns else 0)
        
        # Create columns
        cols = st.columns(columns)
        
        # Distribute fields across columns
        for idx, (field_name, field_spec) in enumerate(field_items):
            col_idx = idx // fields_per_column
            
            with cols[col_idx]:
                field_type = field_spec.get("type", "number")
                extra = field_spec.get("extra", None)

                if field_type == "select":
                    options = field_spec["options"]
                    default_value = field_spec.get("default")
                    default_index = options.index(default_value) if default_value in options else 0
                    
                    payload[field_name] = st.selectbox(
                        field_formater(field_name, extra),
                        options,
                        index=default_index,
                        key=f"{form_key}_{field_name}",
                        help=field_spec.get("help")
                    )
                
                elif field_type == "number":
                    payload[field_name] = st.number_input(
                        field_formater(field_name, extra),
                        min_value=field_spec.get("min", 0),
                        max_value=field_spec.get("max"),
                        value=field_spec.get("default", 0),
                        step=field_spec.get("step", 1),
                        key=f"{form_key}_{field_name}",
                        help=field_spec.get("help")
                    )
                
                elif field_type == "slider":
                    payload[field_name] = st.slider(
                        field_formater(field_name, extra),
                        min_value=field_spec.get("min", 0),
                        max_value=field_spec.get("max", 100),
                        value=field_spec.get("default", 50),
                        step=field_spec.get("step", 1),
                        key=f"{form_key}_{field_name}",
                        help=field_spec.get("help")
                    )
                
                elif field_type == "text":
                    payload[field_name] = st.text_input(
                        field_formater(field_name, extra),
                        value=field_spec.get("default", ""),
                        key=f"{form_key}_{field_name}",
                        help=field_spec.get("help")
                    )
                
                elif field_type == "checkbox":
                    payload[field_name] = st.checkbox(
                        field_formater(field_name, extra),
                        value=field_spec.get("default", False),
                        key=f"{form_key}_{field_name}",
                        help=field_spec.get("help")
                    )
                
                elif field_type == "radio":
                    options = field_spec["options"]
                    default_value = field_spec.get("default")
                    default_index = options.index(default_value) if default_value in options else 0
                    
                    payload[field_name] = st.radio(
                        field_formater(field_name, extra),
                        options,
                        index=default_index,
                        key=f"{form_key}_{field_name}",
                        help=field_spec.get("help")
                    )
        
        submitted = st.form_submit_button(submit_label, use_container_width=True)
    
    return submitted, payload
