# Using MCPs to connect to GUMs

## I just want to set up the MCP

First, you'll need to set up the GUM in general, and have it build some sense of your context. To do this, follow the instructions on [the front page here.](../index.md).

Once you're done with that, just clone the [MCP Repository](https://github.com/GeneralUserModels/gum-mcp) and run the following:

```bash
> git clone git@github.com:GeneralUserModels/gum-mcp.git
> cd gum-mcp
> # maybe create a python environment :)
> pip install --editable .
```

Create a .env file with your environment variables. All you is a user name in the file (e.g.```USER_NAME="Omar Shaikh"```). In sum, your .env file looks something like this:

```bash
USER_NAME="Omar Shaikh"
```

Finally, install the MCP client, pointing to the .env file:

```bash
> mcp install server.py -f .env --with gum-ai
```

The MCP should then be enabled in the Claude app!

## Tutorial

(coming soon: a walkthough on how this was built!)