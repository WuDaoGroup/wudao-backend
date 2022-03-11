from app.routers import api_user, api_file, api_data, api_model, api_explanation, api_causals


prefix = '/api/v1'
def init_routers(app):
    app.include_router(api_user.router, prefix=prefix, tags=["用户管理"])
    app.include_router(api_file.router, prefix=prefix, tags=["文件处理"])
    app.include_router(api_data.router, prefix=prefix, tags=["数据处理"])
    app.include_router(api_model.router, prefix=prefix, tags=["模型相关"])
    app.include_router(api_explanation.router, prefix=prefix, tags=["可解释性"])
    app.include_router(api_causals.router, prefix=prefix, tags=["因果推断"])