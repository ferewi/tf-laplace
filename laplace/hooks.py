from typing import Callable
import tensorflow as tf


def make_hookable(model: tf.keras.models.Sequential):
    """Enable attaching hooks to layers.

    This function extends the layers of the given model about a function
    "register_hook" that allows the attachment of hooks to layers.
    Hooks are callables that allows two things:
        1) Get access to the inputs, pre-activations and activations of the
           respective layers.
        2) Modify the output of the respective layer, if the hook function
           returns something 'not None'.

    A hook definition looks as follows:
    ```python
    my_hook = my_custom_hook(layer: tf.keras.layers.Layer,
                          inputs: tf.Tensor,
                          pre_activation: tf.Tensor,
                          output: tf.Tensor):
        # your custom code
    ```

    If the hook returns a tf.Tensor, the this will be used as the new output
    of the layer. If None is returned by the hook, the "normal" result of the
    "normal" activation of the layer will be used as output.

    To assign a hook to a layer, the ""register_hook" method on the respective
    layer should be called with the layer and the hook as argument:

    ```python
    my_hook = my_custom_hook( ... )
    layer.register_hook(layer, my_hook)
    ```

    Args:
        model: tf.keras.models.Sequential

    Returns:
        void

    """
    for i, layer in enumerate(model.layers):
        layer._hooks = []
        layer.call = _get_call_fn(layer)
        layer.register_hook = lambda l, hook: l._hooks.append({'callable': hook, 'enabled': True})


def disable_hooks(model: tf.keras.models.Sequential):
    """ Disable all hooks that have been registered.

    This might be needed as the hook's dependencies might not be available in
    all stages of training, evaluation or prediction.

    Args:
        model: tf.keras.models.Sequential

    Returns:
        void
    """
    for layer in model.layers:
        try:
            for hook_entry in layer._hooks:
                hook_entry['enabled'] = False
        except AttributeError:
            continue


def enable_hooks(model: tf.keras.models.Sequential):
    """ Enable all hooks that have been registered.

    Re-enable all hooks after they have been disabled.

    Args:
        model: tf.keras.models.Sequential

    Returns:
        void
    """
    for layer in model.layers:
        try:
            for hook_entry in layer._hooks:
                hook_entry['enabled'] = True
        except AttributeError:
            continue


def _get_call_fn(layer: tf.keras.layers.Layer) -> Callable[[tf.Tensor], tf.Tensor]:
    old_call_fn = layer.call
    try:
        old_activation_fn = layer.activation
    except AttributeError:
        old_activation_fn = tf.keras.activations.get('linear')

    def call(input: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        layer.activation = tf.keras.activations.get('linear')

        if old_activation_fn.__name__ is not 'linear':
            pre_activations = old_call_fn(input, *args, **kwargs)
            output = old_activation_fn(pre_activations)
        else:
            pre_activations = old_call_fn(input, *args, **kwargs)
            output = pre_activations

        for hook_entry in layer._hooks:
            if not hook_entry['enabled']:
                continue
            hook = hook_entry['callable']
            hook_result = hook(layer, input, pre_activations, output)
            if hook_result is not None:
                output = hook_result
        return output

    return call
