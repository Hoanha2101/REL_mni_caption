import streamlit as st
import pandas as pd
from server import LOGIC

st.set_page_config(layout="wide")

st.title("Cutting Stock Problem")

RIGHT_LAYOUT, LEFT_LAYOUT = st.columns(2)

METHOD_OPTION = ["First Fit", "Best Fit", "Combination Algorithm"]

# Khởi tạo danh sách trong session_state nếu chưa có
if "stock_data" not in st.session_state:
    st.session_state.stock_data = []
if "required_data" not in st.session_state:
    st.session_state.required_data = []

##########------------------stock panels--------------------##########
with RIGHT_LAYOUT:
    
    choose_method = st.selectbox("Choose inference method:", METHOD_OPTION)
    
    # Nhập dữ liệu về stock panels
    st.subheader("Available stock panels")

    with st.form("stock_input_form"):
        stock_width = st.number_input("Stock Width", min_value=1, value=1, step=1)
        stock_length = st.number_input("Stock Length", min_value=1, value=1, step=1)
        stock_quantity = st.number_input("Stock Quantity", min_value=1, value=1, step=1)

        if st.form_submit_button("Add Stock Panel"):
            st.session_state.stock_data.append({"Width": stock_width, "Length": stock_length, "Quantity": stock_quantity})

    # Hiển thị bảng stock panels đã nhập
    if st.session_state.stock_data:
        st.write("Stock Panels List:")
        df_stock = pd.DataFrame(st.session_state.stock_data)
        df_stock.index = range(1, len(df_stock) + 1)
        st.dataframe(df_stock, use_container_width=True)
        
                # Xóa hàng dựa trên index
        delete_index = st.number_input("Enter index to delete:", min_value=1, max_value=len(df_stock), step=1, key="delete_index_stock") if not df_stock.empty else None

        if st.button("Delete Row stock") and delete_index in df_stock.index:
            st.session_state.stock_data.pop(delete_index - 1)  # Xóa trong danh sách gốc
            st.rerun()  # Làm mới trang để cập nhật DataFrame
        
    ##########-------------------Required panels-------------------##########
    st.subheader("Required panels")

    with st.form("required_input_form"):
        required_width = st.number_input("Required Width", min_value=1, value=1, step=1)
        required_length = st.number_input("Required Length", min_value=1, value=1, step=1)
        required_quantity = st.number_input("Required Quantity", min_value=1, value=1, step=1)

        if st.form_submit_button("Add Required Panel"):
            st.session_state.required_data.append({"Width": required_width, "Length": required_length, "Quantity": required_quantity})

    # Hiển thị bảng required panels đã nhập
    if st.session_state.required_data:
        st.write("Required Panels List:")
        df_required = pd.DataFrame(st.session_state.required_data)
        df_required.index = range(1, len(df_required) + 1)
        st.dataframe(df_required, use_container_width=True)
        
        # Xóa hàng dựa trên index
        delete_index = st.number_input("Enter index to delete:", min_value=1, max_value=len(df_required), step=1, key="delete_index_required") if not df_required.empty else None

        if st.button("Delete Row required") and delete_index in df_required.index:
            st.session_state.required_data.pop(delete_index - 1)  # Xóa trong danh sách gốc
            st.rerun()  # Làm mới trang để cập nhật DataFrame

with LEFT_LAYOUT:
    # Nút chạy thuật toán
    if st.button("Calculate"):
        st.success(f"Running {choose_method} on input stock panels!")
        images = LOGIC(df_stock, df_required, choose_method)  # Gọi hàm trả về danh sách ảnh
        
        # Hiển thị danh sách ảnh
        for img in images:
            st.image(img, use_column_width=True)  # Hiển thị ảnh

