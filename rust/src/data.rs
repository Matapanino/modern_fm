//! Input views and validation shared by FM and FFM kernels.
//!
//! Pure Rust (no pyo3 types) so the math kernels stay unit-testable;
//! errors are `String`s mapped to Python exceptions in `lib.rs`.

/// Borrowed view over a CSR matrix (scipy `indptr`/`indices`/`data` triple).
pub struct CsrView<'a> {
    pub indptr: &'a [i64],
    pub indices: &'a [i64],
    pub data: &'a [f64],
}

impl<'a> CsrView<'a> {
    pub fn new(
        indptr: &'a [i64],
        indices: &'a [i64],
        data: &'a [f64],
        n_cols: usize,
    ) -> Result<Self, String> {
        if indptr.is_empty() {
            return Err("indptr must have length n_rows + 1 (>= 1)".into());
        }
        if indices.len() != data.len() {
            return Err(format!(
                "indices and data length mismatch: {} vs {}",
                indices.len(),
                data.len()
            ));
        }
        if *indptr.last().unwrap() as usize != data.len() || indptr[0] != 0 {
            return Err("indptr must start at 0 and end at nnz".into());
        }
        if indptr.windows(2).any(|p| p[1] < p[0]) {
            return Err("indptr must be non-decreasing".into());
        }
        if indices.iter().any(|&i| i < 0 || i as usize >= n_cols) {
            return Err(format!("column index out of range [0, {n_cols})"));
        }
        Ok(Self {
            indptr,
            indices,
            data,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Nonzero (column indices, values) of row `r`.
    pub fn row(&self, r: usize) -> (&[i64], &[f64]) {
        let lo = self.indptr[r] as usize;
        let hi = self.indptr[r + 1] as usize;
        (&self.indices[lo..hi], &self.data[lo..hi])
    }
}

/// Collect nonzero (index, value) pairs of one dense row, matching the
/// Python reference (`np.nonzero`: exact-zero entries are skipped).
pub fn dense_row_nonzeros(row: &[f64], idx_buf: &mut Vec<usize>, val_buf: &mut Vec<f64>) {
    idx_buf.clear();
    val_buf.clear();
    for (i, &x) in row.iter().enumerate() {
        if x != 0.0 {
            idx_buf.push(i);
            val_buf.push(x);
        }
    }
}

pub fn check_field_ids(field_ids: &[i64], n_features: usize, n_fields: usize) -> Result<(), String> {
    if field_ids.len() != n_features {
        return Err(format!(
            "field_ids length {} does not match n_features {}",
            field_ids.len(),
            n_features
        ));
    }
    if field_ids.iter().any(|&f| f < 0 || f as usize >= n_fields) {
        return Err(format!("field id out of range [0, {n_fields})"));
    }
    Ok(())
}
