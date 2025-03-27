def sankey_plot(df, source, target, value, name='Value'):
    # # Count the occurrences of each unique pair
    # counts = df.groupby([f'{source}',f'{target}']).size().reset_index(name='Value')
    
    # # Create a mapping of unique labels to indices
    # unique_labels = list(set(counts[f'{source}'].tolist() + counts[f'{target}'].tolist()))
    # label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    # # Map source and target to their indices
    # counts['sourceIndex'] = counts[f'{source}'].map(label_to_index)
    # counts['targetIndex'] = counts[f'{target}'].map(label_to_index)
    
    # # Prepare data for Sankey diagram
    # source = counts['sourceIndex'].tolist()
    # target = counts['targetIndex'].tolist()
    # value = counts['Value'].tolist()
    
    # Step 1: Aggregate the data by summing the 'value' for each (source, target) pair
    agg_df = df.groupby([f'{source}', f'{target}'])[f'{value}'].sum().reset_index()
    display(agg_df)
    # Step 2: Create a list of unique sources and targets
    all_nodes = list(set(agg_df[f'{source}']).union(set(agg_df[f'{target}'])))
    
    # Step 3: Create dictionaries to map nodes to indices
    source_indices = {node: i for i, node in enumerate(all_nodes)}
    target_indices = {node: i for i, node in enumerate(all_nodes)}

    # Step 4: Prepare data for the Sankey diagram
    # We need to convert the sources and targets to their corresponding indices
    sankey_data = {
        'source': agg_df[f'{source}'].map(source_indices),
        'target': agg_df[f'{target}'].map(target_indices),
        'value': agg_df[f'{value}']
    }

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=7,
            thickness=20,
            line=dict(color="black", width=0.01),
            label=all_nodes
        ),
        link=dict(
            source=sankey_data['source'],
            target=sankey_data['target'],
            value=sankey_data['value']
        )
    )])
    
    fig.update_layout(
        title_text="Sankey Diagram", font_size=10,
        width=1000,
        height=800    
    )
    fig.show()

    fig.write_html(f'sankey_diagram {name}.html')

def sankey3_diagram(df, source, intermediate, target, value, name='Value'):
    # Step 1: Group the data to aggregate the values for the flows
    df_grouped = df.groupby([source, intermediate, target], as_index=False)[value].sum()
 
    # Step 2: Create a list of unique nodes for all three stages (source, intermediate, target)
    all_nodes = list(set(df_grouped[source]).union(set(df_grouped[intermediate])).union(set(df_grouped[target])))
    # Step 3: Create dictionaries to map nodes to indices
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    # Step 4: Prepare data for the Sankey diagram
    # Map the 'source', 'intermediate', and 'target' fields to their corresponding indices
    df_grouped['source_idx'] = df_grouped[source].map(node_indices)
    df_grouped['intermediate_idx'] = df_grouped[intermediate].map(node_indices)
    df_grouped['target_idx'] = df_grouped[target].map(node_indices)
    # Step 5: Prepare the flow data for the Sankey diagram
    # First, create the flows from source to intermediate
    links_source_intermediate = pd.DataFrame({
        'source': df_grouped['source_idx'],
        'target': df_grouped['intermediate_idx'],
        'value': df_grouped[value]
    })
    # Then, create the flows from intermediate to target
    links_intermediate_target = pd.DataFrame({
        'source': df_grouped['intermediate_idx'],
        'target': df_grouped['target_idx'],
        'value': df_grouped[value]
    })
    # Combine the two sets of links
    all_links = pd.concat([links_source_intermediate, links_intermediate_target], ignore_index=True)
    # Step 6: Create the Sankey diagram using plotly
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.01),
            label=all_nodes
        ),
        link=dict(
            source=all_links['source'],
            target=all_links['target'],
            value=all_links['value']
        )
    ))
    fig.update_layout(
        title_text="Sankey Diagram", font_size=10,
        width=1200,
        height=1000    
    )
    # Show the plot
    fig.show()

    fig.write_html(f'sankey_diagram {name}.html')