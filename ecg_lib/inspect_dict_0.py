def inspect_dict(d, indent=0, max_depth=3):
    prefix = "  " * indent
    
    if indent > max_depth:
        print(prefix + "...")
        return
    
    if isinstance(d, dict):
        print(prefix + f"dict ({len(d)} keys)")
        for k, v in d.items():
            print(prefix + f"  [{k}] -> ", end="")
            inspect_dict(v, indent + 1, max_depth)
    
    elif isinstance(d, (list, tuple)):
        print(prefix + f"{type(d).__name__} (len={len(d)})")
        if len(d) > 0:
            inspect_dict(d[0], indent + 1, max_depth)
    
    else:
        # try to extract useful attributes
        info = type(d).__name__
        
        if hasattr(d, "shape"):
            info += f", shape={d.shape}"
        if hasattr(d, "dtype"):
            info += f", dtype={d.dtype}"
        if hasattr(d, "__len__") and not hasattr(d, "shape"):
            try:
                info += f", len={len(d)}"
            except:
                pass
        
        print(prefix + info)