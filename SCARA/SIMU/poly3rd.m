function y = poly3rd(t, p_i, v_i, t_i, p_f, v_f, t_f)
    b = [p_i; p_f; v_i; v_f];
    A = [t_i^3, t_i^2, t_i, 1;
         t_f^3, t_f^2, t_f, 1;
         3*t_i^2, 2*t_i, 1, 0;
         3*t_f^2, 2*t_f, 1, 0];
    x = inv(A) * b;
    y = x(1)*t^3 + x(2)*t^2 + x(3)*t + x(4);
end
