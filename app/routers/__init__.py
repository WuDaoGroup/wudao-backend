from app.routers import api_user, api_file, api_model


prefix = '/api/v1'
def init_routers(app):
    app.include_router(api_user.router, prefix=prefix, tags=["用户管理"])
    app.include_router(api_file.router, prefix=prefix, tags=["文件处理"])
    app.include_router(api_model.router, prefix=prefix, tags=["模型相关"])