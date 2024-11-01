# Make a sparse_model_matrix for fixest estimate. This only keeps the variables that are not removed from `fixest::feols`
did2s_sparse = function(data, fixest, weights_vector) {
  mat_list = sparse_model_matrix(data, fixest, weights_vector)
  Z = NULL
  
  # Coefficients
  if(!is.null(mat_list$mat_RHS)) {
    Z = mat_list$mat_RHS
    
    select = names(fixest$coefficients)
    cols = colnames(Z)
    
    # Fix i() names
    i_idx = grep("^i\\(", cols)
    cols[i_idx] = sub("^.*__CLEAN__", "", cols[i_idx])
    
    # Subset mat_RHS
    idx = which(select %in% cols)
    Z = Z[,idx]
    
    # If idx is a singular, Z becomes a vector
    if(inherits(Z, "numeric")) Z = Matrix::Matrix(Z, sparse=T)
  }
  
  # Fixed Effects
  if("fixef_id" %in% names(fixest)) {
    
    Z_fixef = mat_list$mat_FE
    
    temp = fixest::fixef(fixest)
    select =	lapply(names(temp), function(var){
      names = names(temp[[var]])
      names = names[temp[[var]] != 0]
      
      paste0(var, "::", names)
    })
    
    # Subset mat_FE
    idx = which(unlist(select) %in% colnames(Z_fixef))
    Z = cbind(Z, Z_fixef[, idx])
  }
  
  return(Z)
}


####
#### Sparse model.matrix ####
#### Written by Laurent Berge
####

# x: fixest estimation
sparse_model_matrix = function(data, x, weights_vector){
  
  # Linear formula
  fml_lin = stats::formula(x, "lin")
  
  
  #
  # Step 1: Linear matrix
  #
  
  vars = attr(stats::terms(fml_lin), "term.labels")
  
  if(length(vars) == 0){
    # Case only FEs
    mat = NULL
  } else {
    
    # Since we don't want to evaluate the factors,
    # the code is a bit intricate because we have to catch them before
    # any interaction takes place
    #
    # that's why I wrap interactions in a function (mult_sparse())
    #
    
    # Below, we evaluate all the variables in a "sparse" way
    
    vars_calls = lapply(vars, mult_wrap)
    
    n = length(vars)
    variables_list = vector("list", n)
    for(i in 1:n){
      variables_list[[i]] = eval(vars_calls[[i]], data)
    }
    
    # To create the sparse matrix, we need the indexes
    
    total_cols = 0
    running_cols = c(0)
    for(i in 1:n){
      xi = variables_list[[i]]
      if(inherits(xi, "sparse_var")){
        total_cols = total_cols + xi$n_cols
      } else {
        total_cols = total_cols + NCOL(xi)
      }
      running_cols[i + 1] = total_cols
    }
    
    # We just create a sparse matrix and fill it
    
    # 1) creating the indexes + names
    
    # NOTA: I use lists to avoid creating copies
    rowid = 1:nrow(data)
    id_all = values_all = names_all = vector("list", n)
    for(i in 1:n){
      xi = variables_list[[i]]
      if(inherits(xi, "sparse_var")){
        id_all[[i]] = cbind(xi$rowid, running_cols[i] + xi$colid)
        values_all[[i]] = xi$values
        names_all[[i]] = paste0(vars[[i]], "::", xi$col_names)
      } else if(NCOL(xi) == 1){
        id_all[[i]] = cbind(rowid, running_cols[i] + 1)
        values_all[[i]] = xi
        names_all[[i]] = vars[[i]]
      } else {
        colid = rep(1:NCOL(xi), each = nrow(data))
        id_all[[i]] = cbind(rep(rowid, NCOL(xi)), running_cols[i] + colid)
        values_all[[i]] = as.vector(xi)
        if(!is.null(colnames(xi))){
          names_all[[i]] = paste0(vars[[i]], colnames(xi))
        } else {
          names_all[[i]] = paste0(vars[[i]], 1:NCOL(xi))
        }
      }
    }
    
    id_mat = do.call(rbind, id_all)
    values_vec = unlist(values_all)
    names_vec = unlist(names_all)
    
    # 2) filling the matrix: one shot, no copies
    
    mat = Matrix::Matrix(0, nrow(data), total_cols, dimnames = list(NULL, names_vec))
    mat[id_mat] = values_vec
  }
  
  #
  # Step 2: the fixed-effects
  #
  
  if(length(x$fixef_id) == 0){
    mat_FE = NULL
  } else {
    # Same process, but easier
    x_full = stats::update(x, .~1, data = data)
    rowid = 1:nrow(data)
    total_cols = sum(x_full$fixef_sizes)
    running_cols = c(0, x_full$fixef_sizes)
    n_FE = length(x_full$fixef_sizes)
    id_all = names_all = vector("list", n_FE)
    
    
    for(i in 1:n_FE){
      xi = x_full$fixef_id[[i]]
      id_all[[i]] = cbind(rowid, running_cols[i] + xi)
      names_all[[i]] = paste0(names(x_full$fixef_id)[i], "::", attr(xi, "fixef_names"))
    }
    
    id_mat = do.call(rbind, id_all)
    names_vec = unlist(names_all)
    
    mat_FE = Matrix::Matrix(0, nrow(data), total_cols, dimnames = list(NULL, names_vec))
    mat_FE[id_mat] = 1
    
  }
  
  
  res = list(mat_RHS = mat, mat_FE = mat_FE)
  
  res
}

# Internal: modifies the calls so that each variable/interaction is evaluated with mult_sparse
mult_wrap = function(x){
  # x: character string of a variable to be evaluated
  # ex: "x1" => mult_sparse(x1)
  #     "x1:factor(x2):x3" => mult_sparse(x3, factor(x2), x1)
  #
  # We also add the argument sparse to i()
  #     "x1:i(species, TRUE)" => mult_sparse(x1, i(species, TRUE, sparse = TRUE))
  
  x_call = str2lang(x)
  
  res = (~ mult_sparse())[[2]]
  
  if(length(x_call) == 1 || x_call[[1]] != ":"){
    res[[2]] = x_call
    
  } else {
    res[[2]] = x_call[[3]]
    tmp = x_call[[2]]
    
    while(length(tmp) == 3 && tmp[[1]] == ":"){
      res[[length(res) + 1]] = tmp[[3]]
      tmp = tmp[[2]]
    }
    
    res[[length(res) + 1]] = tmp
  }
  
  # We also add sparse to i() if found
  for(i in 2:length(res)){
    ri = res[[i]]
    if(length(ri) > 1 && ri[[1]] == "i"){
      ri[["sparse"]] = TRUE
      res[[i]] = ri
    }
  }
  
  if(length(res) > 2){
    # we restore the original order
    res[-1] = rev(res[-1])
  }
  
  return(res)
}

# Internal function to evaluate the variables (and interactions) in a sparse way
mult_sparse = function(...){
  # Only sparsifies factor variables
  # Takes care of interactions
  
  dots = list(...)
  n = length(dots)
  
  num_var = NULL
  factor_list = list()
  info_i = NULL
  is_i = is_factor = FALSE
  # You can't have interactions between i and factors, it's either
  
  for(i in 1:n){
    xi = dots[[i]]
    if(is.numeric(xi)){
      # We stack the product
      num_var = if(is.null(num_var)) xi else xi * num_var
    } else if(inherits(xi, "i_sparse")){
      is_i = TRUE
      info_i = xi
    } else {
      is_factor = TRUE
      factor_list[[length(factor_list) + 1]] = xi
    }
  }
  
  # numeric
  if(!is_i && !is_factor){
    return(num_var)
  }
  # factor()
  if(is_factor){
    factor_list$add_items = TRUE
    factor_list$items.list = TRUE
    
    fact_as_int = do.call(to_integer, factor_list)
    
    values = if(is.null(num_var)) rep(1, length(fact_as_int$x)) else num_var
    
    rowid = seq_along(values)
    res = list(rowid = rowid, colid = fact_as_int$x, values = values,
               col_names = fact_as_int$items, n_cols = length(fact_as_int$items))
    # i()
  } else {
    
    values = info_i$values
    if(!is.null(num_var)){
      num_var = num_var[info_i$rowid]
      values = values * num_var
    }
    
    res = list(rowid = info_i$rowid, colid = info_i$colid,
               values = values[info_i$rowid],
               col_names = info_i$col_names,
               n_cols = length(info_i$col_names))
  }
  
  class(res) = "sparse_var"
  
  res
}