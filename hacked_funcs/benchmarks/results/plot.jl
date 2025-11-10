using CSV, DataFrames
using Plots
using Measures

# 读取数据文件
function plot_results(csv_file::String; title::String="")
    df = CSV.read(csv_file, DataFrame)
    
    # 提取数据并按 sc_target 排序（从大到小）
    df_sorted = sort(df, :sc_target, rev=true)
    sc_target = df_sorted.sc_target
    total_tc_slicing = df_sorted.total_tc_slicing
    total_tc = df_sorted.total_tc
    slice_num_slicing = df_sorted.slice_num_slicing
    slice_num = df_sorted.slice_num
    
    # 创建左边子图：TC 对比
    p1 = plot(sc_target, total_tc_slicing,  # slicing (use original values)
              marker=:circle, linewidth=2.5, markersize=7,
              label="slicing",
              xlabel="sc_target", ylabel="total_tc",
              title="Total TC vs SC Target",
              grid=true, gridwidth=1, gridalpha=0.3,
              legend=:topleft,
              titlefontsize=14, xguidefontsize=14, yguidefontsize=14,
              legendfontsize=12, tickfontsize=11,
              xflip=true,  # 反转横轴，从大到小显示
              bottom_margin=8mm, left_margin=10mm, right_margin=3mm)
    
    plot!(p1, sc_target, total_tc,  # bbtn (use original values)
          marker=:square, linewidth=2.5, markersize=7,
          label="bbtn",
          linestyle=:dash)
    
    # 创建右边子图：Slice Number 对比
    p2 = plot(sc_target, slice_num_slicing,
              marker=:circle, linewidth=2.5, markersize=7,
              label="slicing",
              xlabel="sc_target", ylabel="slice_num",
              title="Slice Number vs SC Target",
              grid=true, gridwidth=1, gridalpha=0.3,
              legend=:topleft,
              yscale=:log10,
              titlefontsize=14, xguidefontsize=14, yguidefontsize=14,
              legendfontsize=12, tickfontsize=11,
              xflip=true,  # 反转横轴，从大到小显示
              bottom_margin=8mm, left_margin=10mm, right_margin=3mm)
    
    plot!(p2, sc_target, slice_num,
          marker=:square, linewidth=2.5, markersize=7,
          label="bbtn",
          linestyle=:dash)
    
    # 组合两个子图
    p = plot(p1, p2, layout=(1, 2), size=(1600, 600), dpi=150)
    
    if !isempty(title)
        plot!(p, title=title, titlefontsize=14)
    end
    
    return p
end

# 比较 bf 和 dfs 的结果
function plot_bf_dfs_comparison(bf_file::String, dfs_file::String; title::String="")
    # 读取两个文件
    df_bf = CSV.read(bf_file, DataFrame)
    df_dfs = CSV.read(dfs_file, DataFrame)
    
    # 提取数据并按 sc_target 排序（从大到小），并过滤 sc_target 在 20-30 之间
    df_bf_filtered = filter(row -> 20 <= row.sc_target <= 30, df_bf)
    df_dfs_filtered = filter(row -> 20 <= row.sc_target <= 30, df_dfs)
    df_bf_sorted = sort(df_bf_filtered, :sc_target, rev=true)
    df_dfs_sorted = sort(df_dfs_filtered, :sc_target, rev=true)
    
    sc_target_bf = df_bf_sorted.sc_target
    total_tc_slicing_bf = df_bf_sorted.total_tc_slicing
    total_tc_bf = df_bf_sorted.total_tc
    slice_num_slicing_bf = df_bf_sorted.slice_num_slicing
    slice_num_bf = df_bf_sorted.slice_num
    
    sc_target_dfs = df_dfs_sorted.sc_target
    total_tc_slicing_dfs = df_dfs_sorted.total_tc_slicing
    total_tc_dfs = df_dfs_sorted.total_tc
    slice_num_slicing_dfs = df_dfs_sorted.slice_num_slicing
    slice_num_dfs = df_dfs_sorted.slice_num
    
    # 创建左边子图：TC 对比
    p1 = plot(sc_target_bf, total_tc_slicing_bf,  # slicing (绿色)
              marker=:circle, linewidth=2.5, markersize=7,
              label="slicing",
              color=:green,
              xlabel="sc_target", ylabel="total_tc",
              title="Total TC vs SC Target",
              grid=true, gridwidth=1, gridalpha=0.3,
              legend=:topleft,
              titlefontsize=14, xguidefontsize=14, yguidefontsize=14,
              legendfontsize=12, tickfontsize=11,
              xflip=true,
              bottom_margin=8mm, left_margin=10mm, right_margin=3mm)
    
    plot!(p1, sc_target_bf, total_tc_bf,  # bbtn_bfs (蓝色，实线)
          marker=:square, linewidth=2.5, markersize=7,
          label="bbtn_bfs",
          color=:blue)
    
    plot!(p1, sc_target_dfs, total_tc_dfs,  # bbtn_dfs (红色，实线)
          marker=:utriangle, linewidth=2.5, markersize=7,
          label="bbtn_dfs",
          color=:red)
    
    # 创建右边子图：Slice Number 对比
    p2 = plot(sc_target_bf, slice_num_slicing_bf,
              marker=:circle, linewidth=2.5, markersize=7,
              label="slicing",
              color=:green,
              xlabel="sc_target", ylabel="slice_num",
              title="Slice Number vs SC Target",
              grid=true, gridwidth=1, gridalpha=0.3,
              legend=:topleft,
              yscale=:log10,
              titlefontsize=14, xguidefontsize=14, yguidefontsize=14,
              legendfontsize=12, tickfontsize=11,
              xflip=true,
              bottom_margin=8mm, left_margin=10mm, right_margin=3mm)
    
    plot!(p2, sc_target_bf, slice_num_bf,
          marker=:square, linewidth=2.5, markersize=7,
          label="bbtn_bfs",
          color=:blue)
    
    plot!(p2, sc_target_dfs, slice_num_dfs,
          marker=:utriangle, linewidth=2.5, markersize=7,
          label="bbtn_dfs",
          color=:red)
    
    # 组合两个子图
    p = plot(p1, p2, layout=(1, 2), size=(1600, 600), dpi=150)
    
    if !isempty(title)
        plot!(p, title=title, titlefontsize=14)
    end
    
    return p
end

# 示例：绘制单个文件
if abspath(PROGRAM_FILE) == @__FILE__

    csv_file = "spin_glass_ground_counting/rrg_n300_d3_ising_dfs.csv"
    
    if isfile(csv_file)
        p = plot_results(csv_file, title="Spin Glass Ground Counting Results")
        savefig(p, replace(csv_file, ".csv" => "_plot.png"))
        println("Plot saved to: ", replace(csv_file, ".csv" => "_plot.png"))
    else
        println("File not found: ", csv_file)
        println("\nAvailable files:")
        for file in readdir("spin_glass_ground_counting", join=true)
            if endswith(file, ".csv")
                println("  ", file)
            end
        end
    end
    
    # 比较 bf 和 dfs 的结果
    # bf_file = "mis_ground_counting/rrg_n300_d3_rand_bf.csv"
    # dfs_file = "mis_ground_counting/rrg_n300_d3_rand_dfs_lp.csv"
    
    # if isfile(bf_file) && isfile(dfs_file)
    #     p = plot_bf_dfs_comparison(bf_file, dfs_file, title="BF vs DFS Comparison")
    #     output_file = replace(bf_file, "_bf.csv" => "_bf_dfs_comparison.png")
    #     savefig(p, output_file)
    #     println("Comparison plot saved to: ", output_file)
    # else
    #     println("Usage example:")
    #     println("  p = plot_bf_dfs_comparison(\"path/to/bf_file.csv\", \"path/to/dfs_file.csv\")")
    #     println("  savefig(p, \"output.png\")")
    # end
end



