---
sidebar_position: 1
hide:
  - navigation
  - toc
---

# General User Models

<div style="text-align: center;">
<img src="fig1.jpg" alt="Figure 1" width="60%" />
</div>

General User Models (GUMs) learn about you by observing _any_ interaction you have with your computer. The GUM takes as input any unstructured observation of a user (e.g., device screenshots) and constructs confidence-weighted propositions that capture the user's knowledge and preferences. GUMs introduce an architecture that infers new propositions about a user from multimodal observations, retrieves related propositions for context, and continuously revises existing propositions.

*tl;dr* Everything you do can be used to make your systems more context-aware.

## Getting Started

First, you'll need to install the GUM package. There are two ways to do it:

!!! info "Getting Started I: Installing the GUM package"

    === "pip"
        Great! Just pip install.

        ```bash
        > pip install -U gum-ai
        ```

    === "From source"

        ```bash
        > git clone git@github.com:GeneralUserModels/gum.git
        > cd gum
        > pip install --editable .
        ```    

You can start a GUM server directly from the command line. 

!!! info "Getting Started II: Starting a GUM server"

    We recommend running this in a tmux or screen session to keep it alive.

    === "Local LMs on a GPU server (recommended)"
        First, install [SGLang](https://sgl-project.github.io/start/install.html) and launch its server with your LM.

        ```bash
        > pip install "sglang[all]"
        > pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ 

        > # Launch the screen VLM model
        > CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server ....


        > # point this to the GUM LM
        > export GUM_LM_API_BASE="base-url"

        > # point this to the VLM
        > export SCREEN_LM_API_BASE="base-url"

        > # point this to the VLM
        > export OPENAI_API_KEY="None"
        ```

        Alternatively, we recommend using [SkyPilot](https://docs.skypilot.co/en/latest/docs/index.html) to serve and run your own models on the cloud. You can use the following [config.yaml]() file in the repo. By default, we use Qwen 2.5 VL 32B (AWQ quanitized). A single H100 (80GB) should give you good enough throughput.

    === "OpenAI"
        You can authenticate by setting the `OPENAI_API_KEY` env variable.

        ```bash
        > export OPENAI_API_KEY="your-api-key-here"
        ```

    Start the GUM listening process up:

    ```bash
    > gum --user-name "Your Name"
    ```

    !!! note "Required Permissions"
        When you first run this command, your system will prompt you to grant accessibility and screen recording permissions to the application. You may need to restart the process a few times as you grant these permissions. This is necessary for GUM to observe your interactions and build its model.

    Once you're all done, go ahead and try querying your GUM to view propositions and observations:

    ```bash
    > gum --query "email"
    ```


## Applications

Once you're all set up, check out the tutorials [here.](tutorials/mcp.md) There are a host of cool applications you can build atop of GUMs.

!!! info "Getting Started III: Querying GUMs with the API"

    One of the main methods you'll use to interface with the GUM is the query function. It's exactly what the CLI calls under the hood. Simply pass your query in as a parameter (uses BM25 under the hood). The query takes many more arguments, which you can read about [here.](api-reference/core.md#gum.gum.gum.query)

    ```python linenums="1"
    import asyncio
    from gum import gum

    gum_instance = gum("Your Name", model="gpt-4.1")

    async def main():
        await gum_instance.connect_db()
        print(await gum_instance.query("email"))

    if __name__ == "__main__":
        asyncio.run(main())
    ```

For example: you can set up [an MCP that uses GUMs here.](tutorials/mcp.md)

## Under the hood

<div style="text-align: center;">
<img src="https://generalusermodels.github.io/final_pipeline.jpg" alt="GUM Pipeline" width="80%" />
</div>

### **Observers** collect raw interaction data.
Observers are modular components that capture various user interactions: screen content, notifications, etc. Each observer operates independently, streaming its observations to the GUM core for processing. We implement a [Screen observer](api-reference/observers.md) as an example.

### **Propositions** describe inferences made about the user.
The core of GUM is its proposition system, which transforms raw observations into structured knowledge. Each proposition carries a confidence score and connects to related information, continuously updating as new evidence arrives.
