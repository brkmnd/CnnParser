# Cnn Parser : C-lang Neural Network Parser

This is a parser that parses C files. You know that programming language from the seventies. The parser is learned with a neural network. It works kind of opposite of a normal parser. Instead of checking whether programs are well formed, it suggests what token type could come next. Instead of stating the grammar formally, the grammar is learned.

- The project is described here https://brkmnd.com/pages/projects/Default.aspx?id=61
- run\_model.py runs the model, it can be trained, validated or tested
- complete\_me.py is a showcase of the parser. It suggests the next syntactical token.
- The parser is done in PyTorch. ~~I can't remember how to detach things from cuda/gpu. So the parser will not run on systems without an nvidia gpu. I'll fix it at some point. Feel free to detach tensors and models yourself~~. Should be fixed :-)
- Since model and dataset each (or together) is not that big, both are included in this repo. The current model in its latest state of training.
- The original C-programs are not.
