# WuDao Data Analysis Platform-backend

<img src="https://pic.imgdb.cn/item/627e1e5a0947543129a2d970.jpg" style="zoom:10%;" />

[![license](https://img.shields.io/badge/release-v1.0.0-<green>)](LICENSE)[![standard-readme compliant](https://img.shields.io/badge/license-MIT-<blue>)](https://github.com/RichardLitt/standard-readme)

​	This is a data analysis visualization cloud platform that is easy to use for the general public.
​	Based on machine learning and causal inference, this data analysis visualization cloud platform can provide one-stop online services for data analysis, including data pre-processing, observation, and prediction tasks such as regression and classification of data using machine learning models, and provide explanations of data or models such as correlation between features, representation of data after dimensionality reduction, and discovery of causal relationships between features.
​	If you want to get the Demo, please click [Demo](https://wudao.netlify.app/).

## Table of Contents

- [Security](#security)
- [Background](#background)
- [Install](#install)
- [API](#api)
- [FEATURES](#FEATURES)
- [Contributing](#contributing)
- [License](#license)

## Security



## Background

​	With the advent of the Big Data era, the field of data analytics has become increasingly important. Various industries such as medical, industrial, and urban can bring scientific discoveries as well as economic benefits with the useful knowledge obtained from the analysis and mining of various types of data. Many people want to learn data analytics skills and use data analytics tools, but often encounter barriers: existing data analytics tools have a high barrier to entry, require programming skills, or have high pricing. This high barrier "dissuades" most potential users. To solve this problem, we want to design and develop a cloud platform for data analysis visualization with low barriers to entry and easy access for the general public.

## Install

```
podman pull tualatinx/wudao-backend
podman run -p 8123:8123 -it tualatinx/wudao-backend:latest bash
uvicorn main:app --host=0.0.0.0 --port=${PORT:-8123} --reload --reload-include='*.py'
```

## API

- Powered by FastAPI, postgresql

## FEATURES

​	This project uses the Python language to write the asynchronous framework FastAPI, performance than Django, Flask and other Python back-end development framework, with high concurrency characteristics. The backend provides the front-end RestfulAPI interface form.

## Contributing

- [LLLeo Li](https://github.com/LLLeoLi)
- [echo17666](https://github.com/echo17666)
- [TualatinX](https://github.com/TualatinX)
- [woarthur](https://github.com/woarthur)
- [DHUAVY](https://github.com/DHUAVY)

## License

[MIT © Richard McRichface.](../LICENSE)



